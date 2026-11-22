"""STAGE1: per-round aggregation diagnostics builder.

One diagnostics row per round, aligned to selected client IDs. Descriptive
only — never additional confirmatory tests.

Policies:
  - Zero-reference norms/cosines become JSON null with an explicit reason.
  - Non-finite evidence raises (never silently replaced by medians).
  - Full-floating-state and parameter-only norms are logged separately.
  - The same-round FedAvg update is computed from the SAME local states
    (no new initialization, no RNG-consuming evaluation).
"""
import math

import numpy as np
import torch


# Method families whose diagnostics rows may carry InfoOpt-specific
# score/constraint/solver fields. Every other arm must export these
# fields as null (protocol requirement); common client-signal summaries
# are logged separately in metrics.csv by train.py regardless.
SOLVER_DIAG_METHODS = frozenset({
    "InfoOpt-Full", "InfoOpt-NoAIB", "InfoOpt-NoMI", "InfoOpt-NoEntropy",
    "InfoOpt-NoJS", "InfoOpt-NoStability", "InfoOpt-NoStop",
    "RA-InfoOpt", "RA-InfoOpt-Heterogeneity", "RA-InfoOpt-Reliability",
})


def diag_export_info(method, info):
    """STAGE1 export policy: return `info` for InfoOpt-family arms and
    None otherwise, so build_round_diag emits null solver/score/shuffle
    fields on non-InfoOpt arms."""
    return info if method in SOLVER_DIAG_METHODS else None


def init_state_hash(model) -> str:
    """STAGE1: canonical content hash of the initial state dict —
    covers tensor names, shapes, dtypes and bytes. Used to REQUIRE
    identical initialization across all six arms of a dataset-seed."""
    import hashlib
    h = hashlib.sha256()
    for name, v in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def enforce_init_identity(shared_dir, dataset, alpha, seed, init_hash,
                          run_dir):
    """STAGE1/3: write-once registry for initial weights. The first arm to
    run records the hash; any later arm whose initial hash differs aborts
    BEFORE local training. The registry is namespaced by
    (dataset, alpha, seed) so heterogeneity regimes never collide.
    Returns the registry path."""
    import json as _json
    from pathlib import Path as _Path
    reg_dir = _Path(shared_dir) / "init_hashes"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg = reg_dir / f"{dataset}_alpha{alpha}_seed{seed}.json"
    payload = {"init_state_sha256": init_hash, "dataset": dataset,
               "alpha": str(alpha), "seed": seed}
    if reg.exists():
        prev = _json.loads(reg.read_text())["init_state_sha256"]
        if prev != init_hash:
            raise RuntimeError(
                f"init-weight mismatch for {dataset} alpha={alpha} "
                f"seed {seed}: this arm's initial weights differ from "
                f"the recorded arm ({reg}); refusing to train")
    else:
        payload["first_arm_run_dir"] = str(run_dir)
        reg.write_text(_json.dumps(payload, indent=1), encoding="utf-8")
    return reg


def _flat(sd, keys=None):
    parts = []
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            continue
        if keys is not None and k not in keys:
            continue
        parts.append(v.detach().cpu().flatten().double())
    return torch.cat(parts) if parts else torch.zeros(1)


def _ratio_cos(actual, ref):
    na = float(actual.norm().item())
    nr = float(ref.norm().item())
    if nr <= 1e-20:
        return None, None, "zero reference norm (same-round FedAvg update)"
    if na <= 1e-20:
        return 0.0, None, "zero actual update norm (cosine undefined)"
    cos = float((actual @ ref).item()) / (na * nr)
    return (na / nr), cos, None


def build_round_diag(round_idx, method, gamma, results, weights, info,
                     global_state, new_state, average_states_fn,
                     param_keys):
    """Build the per-round diagnostics record. Raises on non-finite
    evidence."""
    C = len(results)
    client_ids = [int(r.client_id) for r in results]
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError(f"round {round_idx}: non-finite aggregation "
                         f"weights")
    counts = np.asarray([r.num_samples for r in results], dtype=float)
    if not np.isfinite(counts).all():
        raise ValueError(f"round {round_idx}: non-finite sample counts")
    ref_w = counts / counts.sum() if counts.sum() > 0 else \
        np.full(C, 1.0 / C)

    # weight deviation
    l1 = float(np.abs(w - ref_w).sum())
    l2 = float(np.sqrt(((w - ref_w) ** 2).sum()))
    enc = float(np.exp(-(w * np.log(np.clip(w, 1e-12, None))).sum()))

    # updates from the same local states
    local_states = [r.state for r in results]
    fa_state = average_states_fn(local_states, ref_w.tolist())

    def delta(sd_new, keys=None):
        # detached CPU copies of BOTH operands before subtraction:
        # new_state may be a live GPU state_dict while global_state and
        # client states are CPU snapshots; devices/dtypes are preserved
        # (cpu() copies values without precision change) and inputs are
        # never modified.
        return _flat({k: sd_new[k].detach().cpu()
                      - global_state[k].detach().cpu()
                      for k in global_state.keys()}, keys)

    actual_full = delta(new_state)
    actual_par = delta(new_state, param_keys)
    fa_full = delta(fa_state)
    fa_par = delta(fa_state, param_keys)

    for name, vec in (("actual_full", actual_full),
                      ("actual_params", actual_par), ("fedavg_full", fa_full),
                      ("fedavg_params", fa_par)):
        if not torch.isfinite(vec).all():
            raise ValueError(f"round {round_idx}: non-finite {name} update")

    ratio_f, cos_f, reason_f = _ratio_cos(actual_full, fa_full)
    ratio_p, cos_p, reason_p = _ratio_cos(actual_par, fa_par)

    shuffle = info.get("shuffle_record") if isinstance(info, dict) else None

    return {
        "round": int(round_idx),
        "method": method,
        "gamma": gamma,
        "client_ids": client_ids,
        "weights": [float(x) for x in w.tolist()],
        "sample_size_reference_weights": [float(x) for x in ref_w.tolist()],
        "weight_deviation_l1": l1,
        "weight_deviation_l2": l2,
        "effective_client_number": enc,
        "sample_counts": [int(x) for x in counts.tolist()],
        "scores": info.get("scores") if isinstance(info, dict) else None,
        "shuffle_original_scores": (shuffle.get("original_scores")
                                    if shuffle else None),
        "shuffle_solver_scores": (shuffle.get("solver_scores")
                                  if shuffle else None),
        "shuffle_permutation": (shuffle.get("permutation")
                                if shuffle else None),
        "shuffle_seed_int": (shuffle.get("seed_int") if shuffle else None),
        "update_norm_actual_full_state": float(actual_full.norm().item()),
        "update_norm_actual_params_only": float(actual_par.norm().item()),
        "update_norm_fedavg_full_state": float(fa_full.norm().item()),
        "update_norm_fedavg_params_only": float(fa_par.norm().item()),
        "ratio_actual_over_fedavg_full": ratio_f,
        "cosine_actual_fedavg_full": cos_f,
        "ratio_actual_over_fedavg_params": ratio_p,
        "cosine_actual_fedavg_params": cos_p,
        "null_reason_full": reason_f,
        "null_reason_params": reason_p,
        "temperature_used": (info.get("temperature_used")
                             if isinstance(info, dict) else None),
        "mixture_js": (info.get("mixture_js")
                       if isinstance(info, dict) else None),
        "epsilon": (info.get("epsilon") if isinstance(info, dict) else None),
        "constraint_feasible": (info.get("constraint_feasible")
                                if isinstance(info, dict) else None),
        "constraint_active": (info.get("constraint_active")
                              if isinstance(info, dict) else None),
        "nu": (info.get("nu") if isinstance(info, dict) else None),
        "solver_calls": (info.get("solver_calls")
                         if isinstance(info, dict) else None),
        "solver_final_status": (info.get("solver_final_status")
                                if isinstance(info, dict) else None),
        "fallback_used": (info.get("fallback_used")
                          if isinstance(info, dict) else None),
        "fallback_reason": (info.get("fallback_reason")
                            if isinstance(info, dict) else None),
    }
