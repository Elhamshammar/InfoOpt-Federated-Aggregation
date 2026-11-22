#!/usr/bin/env python3
"""Post-run integrity verifier for the Stage-3 heterogeneity experiment
(six arms, two alpha regimes, hardened for malformed evidence).

Arms: fedavg, uniform, infoopt, shuffled_infoopt, damping_050,
damping_075 — ALL AIB off, enforced in config and checkpoint. Every
check is namespaced by (dataset, alpha, seed); alpha is a required CLI
argument and is verified against the config's partition.alpha.

Checks (structured failure entries, nonzero exit, strict JSON report):
  1. evidence files exist (config_used.json, metrics.csv, summary.json,
     final_model.pt, partition_used.json, diagnostics.jsonl,
     init_state_hash.json);
  2. config matches the arm: dataset, seed, method, use_aib=False,
     uniform_agg flag, gamma value, score_shuffle setting, common
     training settings, early stop disabled, probe disabled;
  3. metrics.csv: rounds exactly 1..100 unique; test_acc finite in [0,1];
     test_loss finite; no early stop triggered;
  4. cohort equals frozen schedule; partition_used equals frozen partition;
  5. checkpoint: loads (weights_only), finite tensors, strict load into
     the recovered model with use_aib=False, NO AIB parameters present;
  6. summary.json early_stop_triggered false;
  7. init_state_hash.json: dataset/seed match the run, hash matches the
     shared initialization registry (cross-arm identity);
  8. diagnostics.jsonl, parsed with STRICT JSON (bare NaN/Infinity
     rejected): exactly 100 rows, rounds exactly 1..100; method/gamma
     match the arm; client_ids equal the frozen cohort; sample_counts
     equal the frozen partition sizes; weights have the cohort length,
     are nonnegative, finite and sum to 1; reference weights, deviations
     and effective client number are recomputed and must match;
     null ratio/cosine requires an explicit reason; non-null cosine must
     lie in [-1, 1]; shuffle seed/permutation/score correspondence is
     rederived exactly on the shuffled arm and must be entirely absent
     elsewhere; constraint/solver fields are type- and
     consistency-checked on the InfoOpt arms and must be null elsewhere;
  9. exporter-key selftest.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

COMMON = {
    "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
    "local_epochs": 2, "batch_size": 64, "clip_grad_norm": 5.0,
    "rounds": 100, "clients_per_round": 5,
}
ARM_RULES = {
    "fedavg":          {"method": "fedavg", "uniform_agg": False,
                        "gamma": None, "score_shuffle": None},
    "uniform":         {"method": "fedavg", "uniform_agg": True,
                        "gamma": None, "score_shuffle": None},
    "infoopt":         {"method": "InfoOpt-Full", "uniform_agg": False,
                        "gamma": None, "score_shuffle": None},
    "shuffled_infoopt": {"method": "InfoOpt-Full", "uniform_agg": False,
                         "gamma": None, "score_shuffle": "stage1"},
    "damping_050":     {"method": "fedavg", "uniform_agg": False,
                        "gamma": 0.5, "score_shuffle": None},
    "damping_075":     {"method": "fedavg", "uniform_agg": False,
                        "gamma": 0.75, "score_shuffle": None},
}
INFOOPT_ARMS = {"infoopt", "shuffled_infoopt"}
DS_NAME = {"cifar100": "CIFAR100", "eurosat": "EuroSAT",
           "pathmnist": "PathMNIST"}
EVIDENCE = ("config_used.json", "metrics.csv", "summary.json",
            "final_model.pt", "partition_used.json", "diagnostics.jsonl",
            "init_state_hash.json")
SHUFFLE_FIELDS = ("shuffle_original_scores", "shuffle_solver_scores",
                  "shuffle_permutation", "shuffle_seed_int")
SOLVER_FIELDS = ("scores", "temperature_used", "mixture_js", "epsilon",
                 "constraint_feasible", "constraint_active", "nu",
                 "solver_calls", "solver_final_status", "fallback_used",
                 "fallback_reason")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_json(text):
    def _reject(x):
        raise ValueError(f"non-strict JSON constant {x!r}")
    return json.loads(text, parse_constant=_reject)


def _shuffle_seed_ref(dataset_name: str, seed: int, round_idx: int) -> int:
    """Independent rederivation of the protocol's shuffle seed."""
    key = f"stage1-score-shuffle-v1|{dataset_name}|{seed}|{round_idx}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8"))
                          .digest()[:8], "big")


BACKBONES = {"smallcnn": "SmallCNN",
             "resnet18": "ResNet18SmallImage",
             "resnet18smallimage": "ResNet18SmallImage"}


def _canon_backbone(name):
    key = str(name).replace("-", "").replace("_", "").lower()
    if key not in BACKBONES:
        raise SystemExit(f"unknown backbone: {name!r} "
                         f"(expected one of {sorted(BACKBONES)})")
    return BACKBONES[key]


def _load_model_ctors(snap):
    import importlib.util
    import sys as _sys
    import types
    pkg = types.ModuleType("snap_models")
    pkg.__path__ = []
    _sys.modules["snap_models"] = pkg
    for mod in ("aib", "cnn", "resnet"):
        spec = importlib.util.spec_from_file_location(
            f"snap_models.{mod}", snap / "models" / f"{mod}.py")
        m = importlib.util.module_from_spec(spec)
        _sys.modules[f"snap_models.{mod}"] = m
        spec.loader.exec_module(m)
    return {"SmallCNN": _sys.modules["snap_models.cnn"].SmallCNN,
            "ResNet18SmallImage":
                _sys.modules["snap_models.resnet"].ResNet18SmallImage}


def _num_classes(dataset):
    return 10 if dataset == "eurosat" else 9 if dataset == "pathmnist"         else 100


def _check_resnet_arch(model, num_classes, failures):
    """Explicit architecture gates for the small-image ResNet-18 variant."""
    import torch.nn as nn
    c = model.conv1
    if not (c.kernel_size == (3, 3) and c.stride == (1, 1)
            and c.padding == (1, 1) and c.out_channels == 64):
        failures.append("resnet stem is not 3x3/stride1/pad1 -> 64")
    if any(isinstance(m, nn.MaxPool2d) for m in model.modules()):
        failures.append("max-pool present (small-image stem forbids it)")
    if not any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
        failures.append("no BatchNorm modules present")
    for i in (1, 2, 3, 4):
        layer = getattr(model, f"layer{i}", None)
        if layer is None or len(layer) != 2:
            failures.append(f"resnet layer{i} does not have 2 blocks")
    if not isinstance(getattr(model, "pool", None), nn.AdaptiveAvgPool2d):
        failures.append("pool is not AdaptiveAvgPool2d")
    if getattr(model.classifier, "out_features", None) != num_classes:
        failures.append(f"classifier out_features != {num_classes}")
    if getattr(model.classifier, "in_features", None) != 512:
        failures.append("classifier in_features != 512")


def _check_diag_row(d, arm, rules, dataset_name, seed, cohorts,
                    part_sizes, failures):
    """Full schema/consistency enforcement for one diagnostics row."""
    r = d.get("round")
    tag = f"round {r}"
    if not isinstance(r, int):
        failures.append("diagnostics row with non-integer round")
        return
    if d.get("method") != rules["method"]:
        failures.append(f"diagnostics method {d.get('method')!r} != "
                        f"{rules['method']!r} at {tag}")
    if d.get("gamma") != rules["gamma"]:
        failures.append(f"diagnostics gamma {d.get('gamma')!r} != "
                        f"{rules['gamma']!r} at {tag}")

    cids = d.get("client_ids")
    counts = d.get("sample_counts")
    w = d.get("weights")
    ref = d.get("sample_size_reference_weights")
    if cohorts is not None and 1 <= r <= len(cohorts):
        if cids != [int(x) for x in cohorts[r - 1]]:
            failures.append(f"diagnostics client_ids != frozen cohort at "
                            f"{tag}")
    if part_sizes is not None and isinstance(cids, list):
        want_counts = [part_sizes.get(str(c)) for c in cids]
        if counts != want_counts:
            failures.append(f"diagnostics sample_counts != partition "
                            f"sizes at {tag}")
    C = len(cids) if isinstance(cids, list) else COMMON[
        "clients_per_round"]
    if not (isinstance(w, list) and len(w) == C
            and all(isinstance(x, (int, float)) and math.isfinite(x)
                    and x >= 0.0 for x in w)):
        failures.append(f"bad weights (length/nonnegativity/finiteness) "
                        f"at {tag}")
        return
    if abs(sum(w) - 1.0) > 1e-9:
        failures.append(f"weights do not sum to 1 at {tag}")
    if not (isinstance(counts, list) and len(counts) == C
            and all(isinstance(x, int) and x > 0 for x in counts)):
        failures.append(f"bad sample_counts at {tag}")
        return
    # recompute reference weights, deviations, ENC
    tot = float(sum(counts))
    ref_exp = [c / tot for c in counts]
    if not (isinstance(ref, list) and len(ref) == C and all(
            abs(a - b) <= 1e-9 for a, b in zip(ref, ref_exp))):
        failures.append(f"sample_size_reference_weights mismatch at {tag}")
    l1 = sum(abs(a - b) for a, b in zip(w, ref_exp))
    l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, ref_exp)))
    enc = math.exp(-sum(x * math.log(max(x, 1e-12)) for x in w))
    for key, want, tol in (("weight_deviation_l1", l1, 1e-9),
                           ("weight_deviation_l2", l2, 1e-9),
                           ("effective_client_number", enc, 1e-6)):
        got = d.get(key)
        if not (isinstance(got, (int, float)) and math.isfinite(got)
                and abs(got - want) <= tol):
            failures.append(f"recomputed {key} mismatch at {tag} "
                            f"(recorded {got!r}, expected {want!r})")
    for key in ("update_norm_actual_full_state",
                "update_norm_actual_params_only",
                "update_norm_fedavg_full_state",
                "update_norm_fedavg_params_only"):
        v = d.get(key)
        if not (isinstance(v, (int, float)) and math.isfinite(v)
                and v >= 0.0):
            failures.append(f"bad {key} at {tag}")
    for ratio_k, cos_k, reason_k in (
            ("ratio_actual_over_fedavg_full", "cosine_actual_fedavg_full",
             "null_reason_full"),
            ("ratio_actual_over_fedavg_params",
             "cosine_actual_fedavg_params", "null_reason_params")):
        cos_v, reason_v = d.get(cos_k), d.get(reason_k)
        ratio_v = d.get(ratio_k)
        if cos_v is None and not (isinstance(reason_v, str)
                                  and reason_v):
            failures.append(f"null {cos_k} without reason at {tag}")
        if cos_v is not None and not (
                isinstance(cos_v, (int, float)) and math.isfinite(cos_v)
                and -1.0 - 1e-9 <= cos_v <= 1.0 + 1e-9):
            failures.append(f"cosine out of [-1,1] at {tag}")
        if ratio_v is not None and not (
                isinstance(ratio_v, (int, float))
                and math.isfinite(ratio_v) and ratio_v >= 0.0):
            failures.append(f"bad {ratio_k} at {tag}")

    # shuffle record: exact on the shuffled arm, absent elsewhere
    sh = {k: d.get(k) for k in SHUFFLE_FIELDS}
    if arm == "shuffled_infoopt":
        if any(v is None for v in sh.values()):
            failures.append(f"incomplete shuffle record at {tag}")
        else:
            want_seed = _shuffle_seed_ref(dataset_name, seed, r)
            if sh["shuffle_seed_int"] != want_seed:
                failures.append(f"shuffle seed mismatch at {tag}")
            perm = sh["shuffle_permutation"]
            orig = sh["shuffle_original_scores"]
            solv = sh["shuffle_solver_scores"]
            if not (isinstance(perm, list) and len(perm) == C
                    and sorted(perm) == list(range(C))):
                failures.append(f"shuffle permutation not an exact "
                                f"permutation at {tag}")
            elif not (isinstance(orig, list) and isinstance(solv, list)
                      and len(orig) == C and len(solv) == C
                      and all(math.isfinite(x) for x in orig + solv)
                      and all(solv[j] == orig[perm[j]]
                              for j in range(C))):
                failures.append(f"shuffle score correspondence broken at "
                                f"{tag}")
    elif any(v is not None for v in sh.values()):
        failures.append(f"shuffle record present on non-shuffled arm at "
                        f"{tag}")

    # constraint/solver fields: validated on InfoOpt arms, null elsewhere
    if arm in INFOOPT_ARMS:
        sc = d.get("scores")
        if not (isinstance(sc, list) and len(sc) == C
                and all(isinstance(x, (int, float)) and math.isfinite(x)
                        for x in sc)):
            failures.append(f"bad scores at {tag}")
        for key in ("temperature_used", "mixture_js", "epsilon", "nu"):
            v = d.get(key)
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                failures.append(f"non-finite {key} at {tag}")
        if not (isinstance(d.get("epsilon"), (int, float))
                and d.get("epsilon") > 0):
            failures.append(f"non-positive epsilon at {tag}")
        for key in ("constraint_feasible", "constraint_active",
                    "fallback_used"):
            if not isinstance(d.get(key), bool):
                failures.append(f"{key} not boolean at {tag}")
        scalls = d.get("solver_calls")
        if not (isinstance(scalls, int) and scalls >= 1):
            failures.append(f"bad solver_calls at {tag}")
        sfs = d.get("solver_final_status")
        if not (isinstance(sfs, dict)
                and isinstance(sfs.get("called"), bool)
                and (sfs.get("success") is None
                     or isinstance(sfs.get("success"), bool))
                and (sfs.get("message") is None
                     or isinstance(sfs.get("message"), str))):
            failures.append(f"bad solver_final_status at {tag}")
        fb_used = d.get("fallback_used")
        fb_reason = d.get("fallback_reason")
        if fb_used is True and not (isinstance(fb_reason, str)
                                    and fb_reason):
            failures.append(f"fallback_used without reason at {tag}")
        if fb_used is False and fb_reason is not None:
            failures.append(f"fallback_reason present without fallback "
                            f"at {tag}")
        if isinstance(sfs, dict) and sfs.get("success") is False \
                and fb_used is not True:
            failures.append(f"solver failure without fallback at {tag}")
    else:
        for key in SOLVER_FIELDS:
            if d.get(key) is not None:
                failures.append(f"solver field {key} present on "
                                f"non-InfoOpt arm at {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alpha", required=True,
                    help="Dirichlet concentration, e.g. 0.1 or 1.0")
    ap.add_argument("--backbone", required=True,
                    help="SmallCNN or ResNet18SmallImage")
    ap.add_argument("--arm", required=True, choices=list(ARM_RULES))
    ap.add_argument("--schedules-dir", required=True)
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    sched_dir = Path(args.schedules_dir)
    alpha = str(float(args.alpha))  # normalize: "1" -> "1.0"
    backbone = _canon_backbone(args.backbone)
    failures = []
    report = {"run_dir": str(rd), "dataset": args.dataset,
              "seed": args.seed, "alpha": str(float(args.alpha)),
              "backbone": backbone,
              "arm": args.arm, "failures": failures,
              "terminal_acc": None}
    rules = ARM_RULES[args.arm]
    dataset_name = DS_NAME[args.dataset]

    for fn in EVIDENCE:
        if not (rd / fn).is_file():
            failures.append(f"missing {fn}")

    # --- frozen shared inputs (used by metrics AND diagnostics checks) ---
    cohorts = None
    sched_path = sched_dir / (f"cohort_schedule_{args.dataset}"
                              f"_alpha{alpha}_seed{args.seed}.json")
    if not sched_path.is_file():
        failures.append(f"frozen schedule missing: {sched_path}")
    else:
        try:
            sp = _strict_json(sched_path.read_text(encoding="utf-8"))
            cohorts = [[int(x) for x in row] for row in sp["schedule"]]
        except Exception as e:
            failures.append(f"malformed frozen schedule: {e}")

    part_sizes = None
    frozen_part = sched_dir / (f"partition_{args.dataset}"
                           f"_alpha{alpha}_seed{args.seed}.json")
    if not frozen_part.is_file():
        failures.append(f"frozen partition missing: {frozen_part}")
    else:
        try:
            pp = _strict_json(frozen_part.read_text(encoding="utf-8"))
            part_sizes = {str(k): len(v)
                          for k, v in pp["client_indices"].items()}
        except Exception as e:
            failures.append(f"malformed frozen partition: {e}")

    # --- config / arm identity ---
    if (rd / "config_used.json").is_file():
        try:
            c = _strict_json((rd / "config_used.json").read_text(
                encoding="utf-8"))
            cfg = c.get("config", c)
            if cfg.get("dataset", {}).get("name") != dataset_name:
                failures.append("dataset mismatch in config")
            if int(cfg.get("seed", -1)) != args.seed:
                failures.append("seed mismatch in config")
            if str(float(cfg.get("partition", {}).get("alpha", "nan"))) \
                    != alpha:
                failures.append(f"partition alpha mismatch in config "
                                f"(expected {alpha})")
            if c.get("method") != rules["method"]:
                failures.append(f"method {c.get('method')!r} != "
                                f"{rules['method']!r}")
            if cfg.get("model", {}).get("use_aib") is not False:
                failures.append("AIB not disabled (stage requires AIB off "
                                "in every arm)")
            if cfg.get("model", {}).get("name") != backbone:
                failures.append(f"model.name {cfg.get('model', {}).get('name')!r} "
                                f"!= expected backbone {backbone!r}")
            tr = cfg.get("training", {})
            for k, v in COMMON.items():
                if tr.get(k) != v:
                    failures.append(f"training.{k}={tr.get(k)} != {v}")
            if bool(tr.get("uniform_agg", False)) != rules["uniform_agg"]:
                failures.append("uniform_agg flag mismatch")
            gamma_saved = c.get("gamma")
            if rules["gamma"] is None:
                if gamma_saved is not None:
                    failures.append(f"gamma {gamma_saved} present on "
                                    f"non-damping arm")
            elif gamma_saved != rules["gamma"]:
                failures.append(f"gamma {gamma_saved} != "
                                f"{rules['gamma']}")
            shuffle_cfg = cfg.get("infoopt", {}).get("score_shuffle")
            if shuffle_cfg != rules["score_shuffle"]:
                failures.append(f"score_shuffle {shuffle_cfg!r} != "
                                f"{rules['score_shuffle']!r}")
            if cfg.get("infoopt", {}).get("early_stop", {}).get(
                    "enabled") not in (False, None):
                failures.append("early stopping not disabled")
            if cfg.get("counterfactual_probe", {}).get("enabled", False):
                failures.append("counterfactual probe not disabled")
        except Exception as e:
            failures.append(f"malformed config_used.json: {e}")

    # --- metrics ---
    rows = []
    if (rd / "metrics.csv").is_file():
        try:
            rows = list(csv.DictReader(open(rd / "metrics.csv",
                                            encoding="utf-8")))
            rset = []
            for r in rows:
                try:
                    rset.append(int(r["round"]))
                except (KeyError, TypeError, ValueError):
                    failures.append(f"malformed round: {r.get('round')}")
            if sorted(rset) != list(range(1, 101)) or \
                    len(rset) != len(set(rset)):
                failures.append("round set is not exactly 1..100 unique")
            for r in rows:
                try:
                    a = float(r["test_acc"])
                    if not math.isfinite(a):
                        failures.append(f"non-finite test_acc round "
                                        f"{r.get('round')}")
                    elif not (0.0 <= a <= 1.0):
                        failures.append(f"test_acc out of [0,1] round "
                                        f"{r.get('round')}")
                except (KeyError, TypeError, ValueError):
                    failures.append(f"malformed test_acc round "
                                    f"{r.get('round')}")
                try:
                    lv = float(r["test_loss"])
                    if not math.isfinite(lv):
                        failures.append(f"non-finite test_loss round "
                                        f"{r.get('round')}")
                except (KeyError, TypeError, ValueError):
                    failures.append(f"malformed test_loss round "
                                    f"{r.get('round')}")
            if any(r.get("early_stop_triggered") == "True" for r in rows):
                failures.append("early_stop_triggered is true")
        except Exception as e:
            failures.append(f"malformed metrics.csv: {e}")

        if cohorts is not None and rows:
            try:
                actual = {}
                for r in rows:
                    actual[int(r["round"])] = [int(x) for x in
                                               r["selected_clients"]
                                               .split(";")]
                got = [actual.get(i + 1) for i in range(len(cohorts))]
                if got != cohorts:
                    mism = [i + 1 for i, (a, b)
                            in enumerate(zip(got, cohorts)) if a != b]
                    failures.append(f"cohort mismatch at rounds {mism[:5]}")
            except Exception as e:
                failures.append(f"schedule comparison failed: {e}")

    # --- partition ---
    if (rd / "partition_used.json").is_file():
        if not frozen_part.is_file():
            pass  # already reported
        else:
            try:
                used = _strict_json((rd / "partition_used.json").read_text(
                    encoding="utf-8"))
                want = _strict_json(frozen_part.read_text(
                    encoding="utf-8"))
                if used != want["client_indices"]:
                    failures.append("partition_used != frozen shared "
                                    "partition")
            except Exception as e:
                failures.append(f"malformed partition evidence: {e}")

    # --- init-weight identity against the shared registry ---
    if (rd / "init_state_hash.json").is_file():
        try:
            ih = _strict_json((rd / "init_state_hash.json").read_text(
                encoding="utf-8"))
            if ih.get("dataset") != dataset_name:
                failures.append("init_state_hash dataset mismatch")
            if int(ih.get("seed", -1)) != args.seed:
                failures.append("init_state_hash seed mismatch")
            if str(ih.get("alpha")) != alpha:
                failures.append("init_state_hash alpha mismatch")
            if str(ih.get("backbone")) != backbone:
                failures.append("init_state_hash backbone mismatch")
            registry = (sched_dir / "init_hashes"
                        / f"{backbone}_{dataset_name}_alpha{alpha}"
                          f"_seed{args.seed}.json")
            if not registry.is_file():
                failures.append(f"init registry missing: {registry}")
            else:
                reg = _strict_json(registry.read_text(encoding="utf-8"))
                if ih.get("init_state_sha256") != \
                        reg.get("init_state_sha256"):
                    failures.append("init hash != shared registry "
                                    "(cross-arm init identity broken)")
        except Exception as e:
            failures.append(f"malformed init evidence: {e}")

    # --- checkpoint (AIB must be ABSENT in every stage1 arm) ---
    if (rd / "final_model.pt").is_file():
        try:
            import torch
            sd = torch.load(rd / "final_model.pt", map_location="cpu",
                            weights_only=True)
            bad_tensors = [k for k, v in sd.items()
                           if torch.is_floating_point(v)
                           and not torch.isfinite(v).all()]
            if bad_tensors:
                failures.append(f"non-finite checkpoint tensors: "
                                f"{bad_tensors[:5]}")
            if any(k.endswith("conv1.3.gamma") for k in sd):
                failures.append("AIB parameters present in checkpoint "
                                "(stage1 requires AIB off)")
            snap = Path(__file__).resolve().parent / "snapshot"
            try:
                ctors = _load_model_ctors(snap)
                model = ctors[backbone](
                    num_classes=_num_classes(args.dataset),
                    input_channels=3, use_aib=False)
                if backbone == "ResNet18SmallImage":
                    _check_resnet_arch(model, _num_classes(args.dataset),
                                       failures)
                model.load_state_dict(sd, strict=True)
            except Exception as e:
                failures.append(f"checkpoint does not load strictly into "
                                f"{backbone}: {str(e)[:200]}")
        except Exception as e:
            failures.append(f"malformed checkpoint: {e}")

    # --- summary ---
    if (rd / "summary.json").is_file():
        try:
            s = _strict_json((rd / "summary.json").read_text(
                encoding="utf-8"))
            if s.get("early_stop_triggered") not in (False, None):
                failures.append("summary early_stop_triggered is true")
        except Exception as e:
            failures.append(f"malformed summary.json: {e}")

    # --- diagnostics.jsonl (strict JSON, full schema enforcement) ---
    if (rd / "diagnostics.jsonl").is_file():
        try:
            diags = [_strict_json(ln) for ln in
                     (rd / "diagnostics.jsonl").read_text(
                         encoding="utf-8").splitlines() if ln.strip()]
            rset = [d.get("round") for d in diags]
            if len(diags) != 100 or rset != list(range(1, 101)):
                failures.append(f"diagnostics rows {len(diags)} / rounds "
                                f"not exactly 1..100")
            for d in diags:
                _check_diag_row(d, args.arm, rules, dataset_name,
                                args.seed, cohorts, part_sizes, failures)
        except Exception as e:
            failures.append(f"malformed diagnostics.jsonl: {e}")

    # --- exporter-key selftest ---
    snap = Path(__file__).resolve().parent / "snapshot"
    try:
        src = (snap / "train.py").read_text(encoding="utf-8")
        if 'metrics["acc"]' not in src:
            failures.append("snapshot train.py does not use "
                            "metrics[\"acc\"]")
    except Exception as e:
        failures.append(f"exporter selftest failed: {e}")

    report["hashes"] = {fn: sha256(rd / fn) for fn in EVIDENCE
                        if (rd / fn).is_file()}
    if not failures and rows:
        try:
            round100 = [r for r in rows if int(r["round"]) == 100]
            if len(round100) == 1:
                a = float(round100[0]["test_acc"])
                report["terminal_acc"] = a if math.isfinite(a) else None
        except Exception:
            report["terminal_acc"] = None
    report["verified"] = not failures

    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, allow_nan=False),
                       encoding="utf-8")
    if failures:
        print(f"VERIFY FAIL {rd.name}: {json.dumps(failures)}",
              file=sys.stderr)
        sys.exit(1)
    print(f"VERIFY OK {rd.name} arm={args.arm} "
          f"terminal_acc={report['terminal_acc']:.4f}")


if __name__ == "__main__":
    main()
