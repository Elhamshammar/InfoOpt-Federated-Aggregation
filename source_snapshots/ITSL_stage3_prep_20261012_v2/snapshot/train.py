from __future__ import annotations
import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from data.datasets import get_dataset, get_labels
from data.partition import dirichlet_partition, class_histogram
from models.build import build_model
from methods.client import train_client
from methods.infoopt import InfoOptController
from methods.ra_infoopt import RAInfoOptController
from methods.feddisco import FedDiscoController
from methods.gpafed import GPAFedController
from utils.aggregation import average_states, apply_server_gradient, apply_magnitude_damping, model_num_bytes, state_delta, weighted_sum_states
from utils.feddyn import feddyn_server_update, init_feddyn_state
from utils.metrics import evaluate
from utils.seed import set_seed
from utils.logger import CSVLogger, save_json
from utils.stage1_diagnostics import (build_round_diag,
                                      init_state_hash,
                                      enforce_init_identity,
                                      diag_export_info)
from utils.counterfactual_probe import (
    create_validation_split,
    create_validation_loader,

    evaluate_probe_candidates,
    log_probe_results,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


INFOOPT_VARIANTS = {
    "InfoOpt-Full",
    "InfoOpt-NoAIB",
    "InfoOpt-NoMI",
    "InfoOpt-NoEntropy",
    "InfoOpt-NoJS",
    "InfoOpt-NoStability",
    "InfoOpt-NoStop",
}

RA_VARIANTS = {
    "RA-InfoOpt",
    "RA-InfoOpt-Heterogeneity",
    "RA-InfoOpt-Reliability",
}

METHOD_CHOICES = [
    "fedavg",
    "fedprox",
    "fednova",
    "scaffold",
    "feddyn",
    "fedopt",
    "ibfl",
    "feddp",
    "feddisco",
    "gpafed",
    "infoopt",
    *sorted(INFOOPT_VARIANTS),
    *sorted(RA_VARIANTS),
]


def jain_fairness(weights: list[float]) -> float:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    den = w.size * np.sum(w ** 2)
    if den <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / den)


def resolve_method(raw: str) -> str:
    if raw == "infoopt":
        return "InfoOpt-Full"
    return raw


def baseline_signal_info(results, weights) -> dict:
    """Baseline per-round signal summary for non-controller arms. Used for
    metrics.csv logging; the diagnostics row receives these fields only
    through diag_export_info(), which nulls them on non-InfoOpt arms."""
    return {
        "mean_mi": float(np.mean([r.mi_proxy for r in results])),
        "mean_entropy": float(np.mean([r.entropy for r in results])),
        "mean_js": 0.0,
        "mean_stability": float(np.mean([-(r.update_norm + r.train_loss_std) for r in results])),
        "mean_score": float(np.mean([r.mi_proxy - r.entropy for r in results])),
        "scores": [float(r.mi_proxy - r.entropy) for r in results],
        "weight_var": float(np.var(weights)),
        "nu": 0.0,
        "epsilon": 0.0,
        "mixture_js": 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, choices=METHOD_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None, help="Override training.rounds (for smoke tests).")
    parser.add_argument("--fedprox-mu", type=float, default=None)
    parser.add_argument("--fedopt-server-lr", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (e.g. 0, 1). Overrides config device.")
    parser.add_argument("--gamma", type=float, default=None, help="Apply magnitude damping gamma to standard FedAvg aggregation (e.g. 0.5, 0.75). Only used with --method fedavg.")
    parser.add_argument("--aib-override", type=str, default=None, choices=["true", "false"],
                        help="FOUR-ARM PATCH: force model.use_aib on/off AFTER the "
                             "method-based AIB defaults. 'true' enables the FedAvg+AIB "
                             "arm; 'false' yields InfoOpt-Full with AIB disabled.")
    parser.add_argument("--score-shuffle", type=str, default=None,
                        choices=["stage1"],
                        help="STAGE1 PATCH: permute InfoOpt's final composite scores "
                             "before the solver (shuffled-score control arm).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.alpha is not None:
        cfg["partition"]["alpha"] = float(args.alpha)
    if args.rounds is not None:
        cfg["training"]["rounds"] = args.rounds

    method = resolve_method(args.method)
    if method not in INFOOPT_VARIANTS and method not in RA_VARIANTS:
        cfg["model"]["use_aib"] = False
    if method == "InfoOpt-NoAIB":
        cfg["model"]["use_aib"] = False
    # FOUR-ARM PATCH: explicit override applied AFTER method-based defaults.
    if args.aib_override is not None:
        cfg["model"]["use_aib"] = (args.aib_override == "true")
    # STAGE1 PATCH: enable the deterministic score-permutation control.
    if args.score_shuffle is not None:
        cfg.setdefault("infoopt", {})["score_shuffle"] = args.score_shuffle

    if args.fedprox_mu is not None:
        cfg["training"]["fedprox_mu"] = float(args.fedprox_mu)
    if args.fedopt_server_lr is not None:
        cfg["training"]["fedopt_server_lr"] = float(args.fedopt_server_lr)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["device"] = device
    else:
        device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    set_seed(int(cfg["seed"]))

    train_set, test_set = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["root"], cfg["dataset"]["num_classes"])
    labels = get_labels(train_set)
    client_indices = dirichlet_partition(
        labels,
        num_clients=cfg["partition"]["num_clients"],
        alpha=cfg["partition"]["alpha"],
        min_samples_per_client=cfg["partition"].get("min_samples_per_client", 10),
        seed=cfg["seed"],
    )

    global_hist = sum(class_histogram(labels, idxs, cfg["dataset"]["num_classes"]) for idxs in client_indices.values())
    model = build_model(cfg).to(device)

    # --- Counterfactual probe model (reusable, never trained) ---
    probe_model = None
    if cfg.get("counterfactual_probe", {}).get("enabled", False):
        probe_model = build_model(cfg).to(device)
    # --- End probe model ---

    test_loader = DataLoader(test_set, batch_size=cfg["evaluation"]["test_batch_size"], shuffle=False, num_workers=2)

    # --- Counterfactual probe: validation split (fixed seed, stratified) ---
    val_indices = None
    val_loader = None
    if cfg.get("counterfactual_probe", {}).get("enabled", False):
        val_seed = int(cfg["seed"]) + 10000
        val_ratio = cfg["counterfactual_probe"].get("val_ratio", 0.15)
        val_indices = create_validation_split(
            labels,
            val_ratio=val_ratio,
            num_classes=cfg["dataset"]["num_classes"],
            seed=val_seed,
        )
        val_loader = create_validation_loader(train_set, val_indices, cfg["training"]["batch_size"])
        print(f"Created validation split: {len(val_indices)} samples (seed={val_seed})")
    # --- End validation split ---

    if method in INFOOPT_VARIANTS:
        controller = InfoOptController(cfg, global_hist, variant=method)
    elif method in RA_VARIANTS:
        controller = RAInfoOptController(cfg, global_hist, variant=method)
    else:
        controller = None

    feddisco_controller = None
    if method == "feddisco":
        feddisco_controller = FedDiscoController(
            num_classes=cfg["dataset"]["num_classes"],
            a=cfg.get("feddisco", {}).get("a", 0.5),
            b=cfg.get("feddisco", {}).get("b", 0.1),
            discrepancy_metric=cfg.get("feddisco", {}).get("discrepancy", "kl"),
        )
        feddisco_controller.precompute_weights(
            client_label_hists={cid: class_histogram(labels, idxs, cfg["dataset"]["num_classes"]) for cid, idxs in client_indices.items()},
            client_num_samples={cid: len(idxs) for cid, idxs in client_indices.items()},
        )

    gpafed_controller = None
    if method == "gpafed":
        gpafed_controller = GPAFedController()

    method_slug = method.replace("-", "_")
    extra_slug = ""
    if method == "fedprox":
        extra_slug = f"_mu{cfg['training'].get('fedprox_mu', 0.0)}"
    if method == "fedopt":
        extra_slug = f"_slr{cfg['training'].get('fedopt_server_lr', 1.0)}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["output"]["dir"]) / f"{cfg['experiment_name']}_{method_slug}{extra_slug}_seed{cfg['seed']}_alpha{cfg['partition']['alpha']}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config": cfg, "method": method, "client_sizes": {str(k): len(v) for k, v in client_indices.items()}, "gamma": args.gamma}, str(run_dir / "config_used.json"))
    # STAGE1 PATCH: canonical initial-weight identity across arms.
    # The hash covers tensor names, shapes, dtypes and bytes. A shared
    # registry (SHARED_DIR/init_hashes) enforces identical initialization
    # across all six arms of each dataset-alpha-seed combination: first
    # arm writes, later arms must match exactly or the run aborts BEFORE
    # local training.
    _alpha = cfg["partition"]["alpha"]
    _init_hash = init_state_hash(model)
    save_json({"init_state_sha256": _init_hash,
               "dataset": cfg["dataset"]["name"],
               "alpha": str(_alpha),
               "seed": int(cfg["seed"])},
              str(run_dir / "init_state_hash.json"))
    _shared_dir = os.environ.get(
        "SHARED_DIR", str(Path(cfg["output"]["dir"]).parent / "shared"))
    enforce_init_identity(_shared_dir, cfg["dataset"]["name"], _alpha,
                          int(cfg["seed"]), _init_hash, run_dir)
    # FOUR-ARM PATCH 2: persist the partition indices actually used so the
    # verifier can prove exact equality with the frozen shared partition.
    save_json({str(k): [int(i) for i in v] for k, v in client_indices.items()},
              str(run_dir / "partition_used.json"))

    logger = CSVLogger(str(run_dir / "metrics.csv"), [


        "round", "method", "test_acc", "test_loss", "precision", "recall",
        "macro_f1", "train_loss", "train_loss_std",
        "mean_mi", "mean_entropy", "mean_js", "mean_stability", "mean_score",
        "weight_var", "temperature_used", "aggregation_entropy", "effective_clients", "effective_clients_exph", "client_fairness",
        "num_selected_clients",
        "model_size_mb", "upload_mb", "download_mb", "metadata_mb", "comm_mb",
        "accuracy_per_mb", "accuracy_per_sec",
        "best_acc", "best_round", "rounds_to_90", "rounds_to_95",
        "early_stop_triggered",
        "round_sec", "elapsed_sec",
        "selected_clients", "weights", "client_scores",
        "nu", "epsilon", "mixture_js",
        # RA-InfoOpt controller metrics
        "heterogeneity_raw", "heterogeneity_normalized",
        "reliability_separation", "reliability_rank", "reliability_temporal", "reliability_total",
        "alpha_raw", "alpha_ema", "alpha_safe",
        "js_fedavg", "js_infoopt", "js_final",
        "enc_fedavg", "enc_infoopt", "enc_final",
        "weight_l1_infoopt_vs_fedavg", "weight_l1_final_vs_fedavg", "fallback_used",
        # Phase 1 instrumentation
        "infoopt_score_std",
        "robust_separation_raw",
        "overlap_count", "rank_rho", "rank_mapped", "rank_valid",
        "score_persistence", "score_persistence_valid",
        "blend_fraction", "blend_identity_error", "blend_identity_error_abs",
        # RA-InfoOpt safety diagnostics (relative-to-FedAvg constraints)
        "js_deviation_limit", "enc_deviation_limit",
        "js_fedavg_deviation", "js_infoopt_deviation", "js_blend_deviation",
        "fedavg_feasible", "infoopt_feasible", "blend_feasible",
        "fallback_reason",
    ])

    # --- Counterfactual probe logger ---
    probe_logger = None
    if cfg.get("counterfactual_probe", {}).get("enabled", False):
        probe_logger = CSVLogger(str(run_dir / "counterfactual_probe.csv"), [
            "round", "delta_val_loss", "delta_val_acc", "fa_val_loss", "fa_val_acc",
            "io_val_loss", "io_val_acc", "weight_l1_diff", "candidate_state_l2_diff",
            "probe_runtime_sec",
        ])
    # --- End probe logger ---

    per_class_f1_log = []

    num_clients = cfg["partition"]["num_clients"]
    clients_per_round = cfg["training"]["clients_per_round"]
    model_bytes = model_num_bytes(model)
    rng = random.Random(cfg["seed"])
    start_time = time.perf_counter()
    best_acc = 0.0
    best_round = 0
    rounds_to_90 = -1
    rounds_to_95 = -1
    upload_mb = 0.0
    download_mb = 0.0
    metadata_mb = 0.0
    model_size_mb = model_bytes / (1024 ** 2)
    early_stop_triggered = False

    scaffold_server_c = None
    scaffold_client_cs = None
    if method == "scaffold":
        scaffold_server_c = {name: torch.zeros_like(p.detach().cpu()) for name, p in model.named_parameters()}
        scaffold_client_cs = {
            cid: {name: torch.zeros_like(p.detach().cpu()) for name, p in model.named_parameters()}
            for cid in range(num_clients)
        }

    fedopt_state = None
    param_keys = {name for name, _ in model.named_parameters()}
    if method == "fedopt":
        fedopt_state = {
            "m": {k: torch.zeros_like(v.detach().cpu()) for k, v in model.state_dict().items() if (torch.is_floating_point(v) and k in param_keys)},
            "v": {k: torch.zeros_like(v.detach().cpu()) for k, v in model.state_dict().items() if (torch.is_floating_point(v) and k in param_keys)},
            "t": 0,
        }

    feddyn_state = None
    if method == "feddyn":
        feddyn_state = init_feddyn_state(
            model, num_clients, float(cfg["training"].get("feddyn_alpha", 0.01))
        )

    global_prototypes = None
    prototype_mb = 0.0
    if method == "feddp":
        feat_dim = int(getattr(getattr(model, "classifier", None), "in_features", 0))
        if feat_dim > 0:
            global_prototypes = torch.zeros(cfg["dataset"]["num_classes"], feat_dim)
            prototype_mb = float(global_prototypes.numel() * 4) / (1024 ** 2)

    for rnd in trange(1, cfg["training"]["rounds"] + 1, desc=f"{method}"):
        round_start = time.perf_counter()
        selected = rng.sample(range(num_clients), clients_per_round)
        results = [
            train_client(
                cid,
                model,
                train_set,
                client_indices[cid],
                cfg,
                device,
                method=method,
                scaffold_server_c=scaffold_server_c,
                scaffold_client_c=(scaffold_client_cs[cid] if scaffold_client_cs is not None else None),
                feddyn_y=(feddyn_state["duals"][cid] if feddyn_state is not None else None),
                global_prototypes=global_prototypes,
                compute_info_signals=True,  # STAGE1: identical signal path in EVERY arm
                round_idx=rnd,              # STAGE1: probe-seed context
            )
            for cid in selected
        ]

        # STAGE1: pre-aggregation global state for diagnostics (clients
        # deep-copy the global model; the model still holds the pre-round
        # state here)
        stage1_global_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
        if method in INFOOPT_VARIANTS:
            weights, info = controller.compute_weights(results, round_idx=rnd)
        elif method in RA_VARIANTS:
            weights, info = controller.compute_weights(results, round_idx=rnd)
        elif method == "feddisco":
            selected_ids = [r.client_id for r in results]
            weights, feddisco_info = feddisco_controller.get_weights(selected_ids)
            info = {
                "mean_mi": float(np.mean([r.mi_proxy for r in results])),
                "mean_entropy": float(np.mean([r.entropy for r in results])),
                "mean_js": 0.0,
                "mean_stability": float(np.mean([-(r.update_norm + r.train_loss_std) for r in results])),
                "mean_score": float(np.mean([r.mi_proxy - r.entropy for r in results])),
                "scores": [float(r.mi_proxy - r.entropy) for r in results],
                "weight_var": float(np.var(weights)),
                "nu": 0.0,
                "epsilon": 0.0,
                "mixture_js": 0.0,
                **feddisco_info,
            }
        elif method == "gpafed":
            global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            local_states = [r.state for r in results]
            sample_counts = [r.num_samples for r in results]
            weights_np, gpafed_info = gpafed_controller.compute_weights(global_state, local_states, sample_counts)
            weights = weights_np.tolist()
            info = {
                "mean_mi": float(np.mean([r.mi_proxy for r in results])),
                "mean_entropy": float(np.mean([r.entropy for r in results])),
                "mean_js": 0.0,
                "mean_stability": float(np.mean([-(r.update_norm + r.train_loss_std) for r in results])),
                "mean_score": float(np.mean([r.mi_proxy - r.entropy for r in results])),
                "scores": [float(r.mi_proxy - r.entropy) for r in results],
                "weight_var": float(np.var(weights)),
                "nu": 0.0,
                "epsilon": 0.0,
                "mixture_js": 0.0,
                **gpafed_info,
            }
        else:
            if cfg["training"].get("uniform_agg", False):
                weights = [1.0 / len(results)] * len(results)
            else:
                ns = np.array([r.num_samples for r in results], dtype=float)
                weights = (ns / ns.sum()).tolist()
            info = baseline_signal_info(results, weights)

        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if method == "fednova":
            deltas = [state_delta(global_state, r.state) for r in results]
            norm_deltas = []
            for d, r in zip(deltas, results):
                k = max(int(r.local_steps), 1)
                nd = {name: (tensor / k if torch.is_floating_point(tensor) else tensor) for name, tensor in d.items()}
                norm_deltas.append(nd)
            agg_norm = weighted_sum_states(norm_deltas, weights)
            avg_steps = float(np.sum(np.asarray(weights) * np.asarray([max(r.local_steps, 1) for r in results], dtype=float)))
            new_state = {
                name: (global_state[name] + avg_steps * agg_norm[name]) if torch.is_floating_point(global_state[name]) else global_state[name]
                for name in global_state.keys()
            }
            model.load_state_dict(new_state, strict=True)
        elif method == "fedopt":
            avg_state = average_states([r.state for r in results], weights)
            grad_state = {
                name: (global_state[name] - avg_state[name])
                for name in param_keys
            }
            fedopt_state["t"] += 1
            beta1 = float(cfg["training"].get("fedopt_beta1", 0.9))
            beta2 = float(cfg["training"].get("fedopt_beta2", 0.99))
            eps = float(cfg["training"].get("fedopt_eps", 1e-8))
            server_lr = float(cfg["training"].get("fedopt_server_lr", 1.0))
            adam_grad = {}
            for name, g in grad_state.items():
                if not torch.is_floating_point(g):
                    continue
                m = fedopt_state["m"][name] = beta1 * fedopt_state["m"][name] + (1 - beta1) * g
                v = fedopt_state["v"][name] = beta2 * fedopt_state["v"][name] + (1 - beta2) * (g * g)
                m_hat = m / (1 - beta1 ** fedopt_state["t"])
                v_hat = v / (1 - beta2 ** fedopt_state["t"])
                adam_grad[name] = m_hat / (torch.sqrt(v_hat) + eps)
            apply_server_gradient(model, adam_grad, server_lr)
            if avg_state is not None:
                state_after = model.state_dict()
                for name in state_after.keys():
                    if name not in param_keys:
                        state_after[name] = avg_state[name]
                model.load_state_dict(state_after, strict=True)
        elif method == "feddyn" and feddyn_state is not None:
            new_state = feddyn_server_update(
                feddyn_state["duals"], results, global_state
            )
            model.load_state_dict(new_state, strict=True)
        else:
            new_state = average_states([r.state for r in results], weights)
            # Apply magnitude damping if gamma is specified (fixed-gamma control)
            if args.gamma is not None and method == "fedavg":
                model.load_state_dict(
                    apply_magnitude_damping(new_state, global_state,
                                            float(args.gamma)),
                    strict=True)
            else:
                model.load_state_dict(new_state, strict=True)

        # STAGE1: per-round diagnostics row (descriptive only)
        if "new_state" in dir() or True:
            _new_state = model.state_dict()
            try:
                # STAGE1 (repair2): InfoOpt-specific score/constraint/
                # solver fields are exported only on InfoOpt-family arms;
                # all other arms write nulls. `info` itself is unchanged
                # and still feeds metrics.csv logging below.
                _diag = build_round_diag(
                    rnd, method,
                    (float(args.gamma) if args.gamma is not None else None),
                    results, weights, diag_export_info(method, info),
                    stage1_global_state,
                    _new_state, average_states, param_keys)
                with open(run_dir / "diagnostics.jsonl", "a",
                          encoding="utf-8") as _df:
                    _df.write(json.dumps(_diag, allow_nan=False)
                              + chr(10))
            except ValueError as _e:
                raise RuntimeError(f"stage1 diagnostics: {_e}")

        if method == "feddp" and global_prototypes is not None:
            total_counts = torch.zeros(global_prototypes.size(0))
            proto_sum = torch.zeros_like(global_prototypes)
            for r in results:
                if r.local_prototypes is None or r.local_proto_counts is None:
                    continue
                counts = r.local_proto_counts.detach().cpu()
                total_counts += counts
                proto_sum += r.local_prototypes.detach().cpu() * counts.unsqueeze(1)
            den = total_counts.unsqueeze(1).clamp_min(1.0)
            global_prototypes = proto_sum / den

        if method == "scaffold":
            for r in results:
                scaffold_client_cs[r.client_id] = r.scaffold_ci_next
            deltas = weighted_sum_states([r.scaffold_ci_delta for r in results], [1.0 / len(results)] * len(results))
            factor = float(len(results)) / float(num_clients)
            for name in scaffold_server_c.keys():
                scaffold_server_c[name] = scaffold_server_c[name] + factor * deltas[name]

        upload_mb += model_bytes / (1024 ** 2) * len(selected)
        download_mb += model_bytes / (1024 ** 2) * len(selected)
        metadata_mb += (model_bytes / (1024 ** 2)) * 0.01 * len(selected) if (method in INFOOPT_VARIANTS or method in RA_VARIANTS) else 0.0
        metadata_mb += 2.0 * prototype_mb * len(selected) if method == "feddp" else 0.0
        cumulative_comm_mb = upload_mb + download_mb + metadata_mb


        # --- Counterfactual probe (Phase 1: log only, no behavior change) ---
        if (
            cfg.get("counterfactual_probe", {}).get("enabled", False)
            and val_loader is not None
            and probe_model is not None
            and method in RA_VARIANTS
            and rnd % cfg["counterfactual_probe"].get("probe_every", 10) == 0
        ):
            probe_start = time.perf_counter()
            # Save RNG state
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None

            # Build FedAvg and InfoOpt candidates from SAME global state and SAME results
            global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            local_states = [r.state for r in results]
            sample_counts = [r.num_samples for r in results]

            # FedAvg weights
            ns = np.array(sample_counts, dtype=float)
            fa_weights = (ns / ns.sum()).tolist()

            # InfoOpt weights from controller
            io_weights = controller.get_last_weights()
            if io_weights is None:
                io_weights = fa_weights

            fa_state = average_states(local_states, fa_weights)
            io_state = average_states(local_states, io_weights)

            # Evaluate both on validation set
            fa_metrics, io_metrics = evaluate_probe_candidates(
                probe_model, fa_state, io_state, val_loader, device
            )

            # Compute diagnostics
            weight_l1_diff = float(np.sum(np.abs(np.array(io_weights) - np.array(fa_weights))))
            candidate_state_l2 = 0.0
            for k in fa_state.keys():
                if torch.is_floating_point(fa_state[k]): candidate_state_l2 += torch.norm(fa_state[k] - io_state[k], p=2).item() ** 2
            candidate_state_l2 = candidate_state_l2 ** 0.5

            log_probe_results(
                probe_logger, rnd, fa_metrics, io_metrics,
                weight_l1_diff, candidate_state_l2, time.perf_counter() - probe_start,
            )

            # Restore RNG state (guarantees no training side effects)
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
        # --- End counterfactual probe ---

        if rnd % cfg["evaluation"].get("eval_every", 1) == 0:
            metrics = evaluate(model, test_loader, device)
            train_loss = float(np.mean([r.train_loss for r in results]))
            train_loss_std = float(np.mean([r.train_loss_std for r in results]))
            elapsed = time.perf_counter() - start_time
            round_sec = time.perf_counter() - round_start

            acc = metrics["acc"]
            if acc > best_acc:
                best_acc = acc
                best_round = rnd
            if rounds_to_90 == -1 and acc >= 0.90:
                rounds_to_90 = rnd
            if rounds_to_95 == -1 and acc >= 0.95:
                rounds_to_95 = rnd



            ws = np.asarray(weights, dtype=float)
            aggregation_entropy = float(-np.sum(ws * np.log(ws + 1e-12)))
            effective_clients = float(1.0 / np.sum(ws ** 2))
            effective_clients_exph = float(np.exp(aggregation_entropy))
            acc_per_mb = acc / max(cumulative_comm_mb, 1e-12)
            acc_per_sec = acc / max(elapsed, 1e-12)

            logger.log({
                "round": rnd,
                "method": method,
                "test_acc": acc,
                "test_loss": metrics["loss"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
                "train_loss": train_loss,
                "train_loss_std": train_loss_std,
                "mean_mi": info["mean_mi"],
                "mean_entropy": info["mean_entropy"],
                "mean_js": info.get("mean_js", 0.0),
                "mean_stability": info.get("mean_stability", 0.0),
                "mean_score": info.get("mean_score", 0.0),
                "weight_var": info["weight_var"],
                "temperature_used": info.get("temperature_used", float(cfg["infoopt"].get("temperature", 1.0))),
                "aggregation_entropy": aggregation_entropy,
                "effective_clients": effective_clients,
                "effective_clients_exph": effective_clients_exph,
                "client_fairness": jain_fairness(weights),
                "num_selected_clients": len(selected),
                "model_size_mb": model_size_mb,
                "upload_mb": upload_mb,
                "download_mb": download_mb,
                "metadata_mb": metadata_mb,
                "comm_mb": cumulative_comm_mb,
                "accuracy_per_mb": acc_per_mb,
                "accuracy_per_sec": acc_per_sec,
                "best_acc": best_acc,
                "best_round": best_round,
                "rounds_to_90": rounds_to_90,
                "rounds_to_95": rounds_to_95,
                "early_stop_triggered": str(early_stop_triggered),
                "round_sec": round_sec,
                "elapsed_sec": elapsed,
                "selected_clients": ";".join(map(str, selected)),
                "weights": ";".join(f"{w:.4f}" for w in weights),
                "client_scores": ";".join(f"{s:.4f}" for s in info.get("scores", [])),
                "nu": info.get("nu", 0.0),
                "epsilon": info.get("epsilon", 0.0),
                "mixture_js": info.get("mixture_js", 0.0),
                "heterogeneity_raw": info.get("heterogeneity_raw", 0.0),
                "heterogeneity_normalized": info.get("heterogeneity_normalized", 0.0),
                "reliability_separation": info.get("reliability_separation", 0.0),
                "reliability_rank": info.get("reliability_rank", 0.0),
                "reliability_temporal": info.get("reliability_temporal", 0.0),
                "reliability_total": info.get("reliability_total", 0.0),
                "alpha_raw": info.get("alpha_raw", 0.0),
                "alpha_ema": info.get("alpha_ema", 0.0),
                "alpha_safe": info.get("alpha_safe", 0.0),
                "js_fedavg": info.get("js_fedavg", 0.0),
                "js_infoopt": info.get("js_infoopt", 0.0),
                "fallback_used": info.get("fallback_used", 0),
                "infoopt_score_std": info.get("infoopt_score_std", 0.0),
                "robust_separation_raw": info.get("robust_separation_raw", 0.0),
                "overlap_count": info.get("overlap_count", 0),
                "rank_rho": info.get("rank_rho", float("nan")),
                "rank_mapped": info.get("rank_mapped", float("nan")),
                "rank_valid": info.get("rank_valid", 0),
                "score_persistence": info.get("score_persistence", float("nan")),
                "score_persistence_valid": info.get("score_persistence_valid", 0),
                "blend_fraction": info.get("blend_fraction", float("nan")),
                "blend_identity_error": info.get("blend_identity_error", 0.0),
                "blend_identity_error_abs": info.get("blend_identity_error_abs", 0.0),
                "enc_fedavg": info.get("enc_fedavg", 0.0),
                "enc_infoopt": info.get("enc_infoopt", 0.0),
                "enc_final": info.get("enc_final", 0.0),
                "weight_l1_infoopt_vs_fedavg": info.get("weight_l1_infoopt_vs_fedavg", 0.0),
                "weight_l1_final_vs_fedavg": info.get("weight_l1_final_vs_fedavg", 0.0),
                "fallback_used": info.get("fallback_used", 0),
                "js_deviation_limit": info.get("js_deviation_limit", 0.0),
                "enc_deviation_limit": info.get("enc_deviation_limit", 0.0),
                "js_fedavg_deviation": info.get("js_fedavg_deviation", 0.0),
                "js_infoopt_deviation": info.get("js_infoopt_deviation", 0.0),
                "js_blend_deviation": info.get("js_blend_deviation", 0.0),
                "fedavg_feasible": info.get("fedavg_feasible", 1),
                "infoopt_feasible": info.get("infoopt_feasible", 0),
                "blend_feasible": info.get("blend_feasible", 0),
                "fallback_reason": info.get("fallback_reason", "none"),
            })
            print(
                f"round={rnd} method={method} acc={acc:.4f} f1={metrics['macro_f1']:.4f} "
                f"loss={metrics['loss']:.4f} comm_mb={cumulative_comm_mb:.2f} best_acc={best_acc:.4f}",
                flush=True,
            )
            per_class_f1_log.append({"round": rnd, "per_class_f1": metrics["per_class_f1"]})

            if (method in INFOOPT_VARIANTS or method in RA_VARIANTS) and controller.should_stop(rnd, metrics["acc"], info):
                print(f"Early stopping at round {rnd}")
                early_stop_triggered = True
                break

    logger.close()
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    save_json({"early_stop_triggered": early_stop_triggered, "best_acc": best_acc, "best_round": best_round}, str(run_dir / "summary.json"))
    if per_class_f1_log:
        save_json(per_class_f1_log, str(run_dir / "per_class_f1_per_round.json"))
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
