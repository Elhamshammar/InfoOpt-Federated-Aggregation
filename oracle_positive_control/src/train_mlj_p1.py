#!/usr/bin/env python3
"""Isolated GPU runner for the MLJ-P1 oracle-corruption positive-control.

This script reuses the historical InfoOpt Stage-3 training pipeline (read-only)
from ``InfoOpt_Code_For_Journal/source_snapshots/ITSL_stage3_prep_20261012_v2``.
It adds deterministic label corruption, oracle signal injection, and the
InfoOpt-NoJS weighting variant configured for the positive control.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import trange


ROOT = Path(__file__).resolve().parents[1]
# Prefer a packaged historical snapshot; fall back to the original development
# location so local Windows dry-runs continue to work without copying.
SNAPSHOT_ROOT = (
    ROOT / "historical_snapshot"
    if (ROOT / "historical_snapshot").exists()
    else ROOT.parent
    / "InfoOpt_Code_For_Journal"
    / "source_snapshots"
    / "ITSL_stage3_prep_20261012_v2"
    / "snapshot"
)
sys.path.insert(0, str(SNAPSHOT_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from corruption_manifest import (  # noqa: E402
    client_corruption_map,
    load_manifest,
    select_corrupted_clients,
)
from corrupted_dataset import CorruptedDataset  # noqa: E402
from data.datasets import get_dataset, get_labels  # noqa: E402
from data.partition import class_histogram, dirichlet_partition  # noqa: E402
from methods.client import train_client  # noqa: E402
from methods.infoopt import InfoOptController  # noqa: E402
from models.build import build_model  # noqa: E402
from utils.aggregation import average_states, model_num_bytes  # noqa: E402
from utils.logger import CSVLogger, save_json  # noqa: E402
from utils.metrics import evaluate  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.stage1_diagnostics import (  # noqa: E402
    build_round_diag,
    diag_export_info,
    enforce_init_identity,
    init_state_hash,
)


METHOD_CHOICES = ["fedavg", "oracle_nojs_aligned", "oracle_nojs_shuffled"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def jain_fairness(weights: list[float]) -> float:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    den = w.size * np.sum(w ** 2)
    if den <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / den)


def baseline_signal_info(results, weights) -> dict:
    return {
        "mean_mi": float(np.mean([r.mi_proxy for r in results])),
        "mean_entropy": float(np.mean([r.entropy for r in results])),
        "mean_js": 0.0,
        "mean_stability": float(
            np.mean([-(r.update_norm + r.train_loss_std) for r in results])
        ),
        "mean_score": float(np.mean([r.mi_proxy - r.entropy for r in results])),
        "scores": [float(r.mi_proxy - r.entropy) for r in results],
        "weight_var": float(np.var(weights)),
        "nu": 0.0,
        "epsilon": 0.0,
        "mixture_js": 0.0,
    }


def configure_infoopt_for_oracle(cfg: dict, shuffled: bool = False) -> None:
    """Set InfoOpt-NoJS parameters used by the oracle arms."""
    io = cfg.setdefault("infoopt", {})
    io["lambda_mi"] = 1.0
    io["lambda_entropy"] = 0.0
    io["lambda_stability"] = 0.0
    io["signal_ema_beta"] = 1.0
    io["robust_scale"] = False
    io["kappa_frac"] = 0.6
    io["temperature"] = 1.0
    io["js_mode"] = "constraint"
    io["score_clip"] = None
    io["fedavg_mix"] = 0.0
    io["min_weight"] = 0.0
    io["min_weight_fraction"] = 0.0
    io.setdefault("early_stop", {})
    io["early_stop"]["enabled"] = False
    io["epsilon_rel"] = 1.0
    io["normalize_mi"] = True
    io["probe_size"] = 256
    io["probe_batch_size"] = 256
    if shuffled:
        io["score_shuffle"] = "stage1"
    else:
        io.pop("score_shuffle", None)


def resolve_data_root(cfg: dict, root: Path = ROOT) -> str:
    """Use configured dataset root if it exists, otherwise fall back locally."""
    configured = Path(os.environ.get("DATASETS_ROOT", cfg["dataset"]["root"]))
    if configured.exists():
        return str(configured)
    fallback = root / "_dataset_cache"
    if not fallback.exists():
        raise RuntimeError(f"Dataset root missing: neither {configured} nor {fallback}")
    return str(fallback)


def make_run_dir(method: str, seed: int, alpha: float) -> Path:
    # Use microsecond precision to avoid collisions when runs are launched in
    # rapid succession (e.g., from a shell loop).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "runs" / f"mlj_p1_{method}_seed{seed}_alpha{alpha}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _sanitize_for_json(obj: Any) -> Any:
    """Replace non-finite floats with None for JSON serialization."""
    import math

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


METRIC_COLUMNS = [
    "round",
    "method",
    "test_acc",
    "test_loss",
    "precision",
    "recall",
    "macro_f1",
    "train_loss",
    "train_loss_std",
    "mean_mi",
    "mean_entropy",
    "mean_js",
    "mean_stability",
    "mean_score",
    "weight_var",
    "temperature_used",
    "aggregation_entropy",
    "effective_clients",
    "effective_clients_exph",
    "client_fairness",
    "num_selected_clients",
    "model_size_mb",
    "upload_mb",
    "download_mb",
    "metadata_mb",
    "comm_mb",
    "accuracy_per_mb",
    "accuracy_per_sec",
    "best_acc",
    "best_round",
    "rounds_to_90",
    "rounds_to_95",
    "early_stop_triggered",
    "round_sec",
    "elapsed_sec",
    "selected_clients",
    "weights",
    "client_scores",
    "nu",
    "epsilon",
    "mixture_js",
    "fallback_used",
    "fallback_reason",
    # RA-InfoOpt columns kept for schema compatibility; always zero here.
    "heterogeneity_raw",
    "heterogeneity_normalized",
    "reliability_separation",
    "reliability_rank",
    "reliability_temporal",
    "reliability_total",
    "alpha_raw",
    "alpha_ema",
    "alpha_safe",
    "js_fedavg",
    "js_infoopt",
    "js_final",
    "enc_fedavg",
    "enc_infoopt",
    "enc_final",
    "weight_l1_infoopt_vs_fedavg",
    "weight_l1_final_vs_fedavg",
    "infoopt_score_std",
    "robust_separation_raw",
    "overlap_count",
    "rank_rho",
    "rank_mapped",
    "rank_valid",
    "score_persistence",
    "score_persistence_valid",
    "blend_fraction",
    "blend_identity_error",
    "blend_identity_error_abs",
    "js_deviation_limit",
    "enc_deviation_limit",
    "js_fedavg_deviation",
    "js_infoopt_deviation",
    "js_blend_deviation",
    "fedavg_feasible",
    "infoopt_feasible",
    "blend_feasible",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--method", type=str, choices=METHOD_CHOICES, required=True
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.alpha is not None:
        cfg["partition"]["alpha"] = float(args.alpha)
    if args.rounds is not None:
        cfg["training"]["rounds"] = args.rounds

    # Ensure AIB is disabled for all MLJ-P1 arms.
    cfg.setdefault("model", {})
    cfg["model"]["use_aib"] = False

    is_oracle = args.method.startswith("oracle_nojs")
    if is_oracle:
        configure_infoopt_for_oracle(
            cfg, shuffled=(args.method == "oracle_nojs_shuffled")
        )
        controller_method = "InfoOpt-NoJS"
    else:
        controller_method = "fedavg"

    # Dataset root: fall back to the local cache for Windows dry-runs.
    cfg["dataset"]["root"] = resolve_data_root(cfg)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["device"] = device
    else:
        device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    seed = int(cfg["seed"])
    alpha = float(cfg["partition"]["alpha"])
    set_seed(seed)

    train_set, test_set = get_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["root"],
        cfg["dataset"]["num_classes"],
    )
    labels = get_labels(train_set)
    num_classes = cfg["dataset"]["num_classes"]

    # Load deterministic corruption manifest for this alpha/seed.
    manifest = load_manifest(alpha=alpha, seed=seed)
    corrupted_clients = manifest["corrupted_clients"]
    corruption_map: dict[int, int] = {}
    for cid in corrupted_clients:
        corruption_map.update(client_corruption_map(manifest, cid))
    train_set = CorruptedDataset(train_set, corruption_map)

    # Build the exact Stage-3 partition.
    client_indices = dirichlet_partition(
        labels,
        num_clients=cfg["partition"]["num_clients"],
        alpha=alpha,
        min_samples_per_client=cfg["partition"].get(
            "min_samples_per_client", 50
        ),
        seed=seed,
    )
    global_hist = sum(
        class_histogram(labels, idxs, num_classes)
        for idxs in client_indices.values()
    )

    model = build_model(cfg).to(device)
    param_keys = {name for name, _ in model.named_parameters()}
    model_bytes = model_num_bytes(model)

    run_dir = make_run_dir(args.method, seed, alpha)

    # Initial-model hash identity across arms.
    init_hash = init_state_hash(model)
    save_json(
        {
            "init_state_sha256": init_hash,
            "dataset": cfg["dataset"]["name"],
            "alpha": str(alpha),
            "seed": seed,
        },
        str(run_dir / "init_state_hash.json"),
    )
    shared_dir = ROOT / "shared"
    enforce_init_identity(
        shared_dir, cfg["dataset"]["name"], alpha, seed, init_hash, run_dir
    )

    # Persist partition and config.
    save_json(
        {str(k): [int(i) for i in v] for k, v in client_indices.items()},
        str(run_dir / "partition_used.json"),
    )

    # Historical participation schedule.
    num_clients = cfg["partition"]["num_clients"]
    clients_per_round = cfg["training"]["clients_per_round"]
    rounds = cfg["training"]["rounds"]
    rng = random.Random(seed)
    schedule = [
        rng.sample(range(num_clients), clients_per_round)
        for _ in range(rounds)
    ]

    save_json(
        {
            "config": cfg,
            "method": args.method,
            "controller_method": controller_method,
            "seed": seed,
            "alpha": alpha,
            "rounds": rounds,
            "client_sizes": {
                str(k): len(v) for k, v in client_indices.items()
            },
            "corrupted_clients": corrupted_clients,
            "schedule": schedule,
        },
        str(run_dir / "config_used.json"),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg["evaluation"]["test_batch_size"],
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 2)),
    )

    controller = None
    if is_oracle:
        controller = InfoOptController(cfg, global_hist, variant="InfoOpt-NoJS")

    logger = CSVLogger(str(run_dir / "metrics.csv"), METRIC_COLUMNS)

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
    per_class_f1_log: list[dict] = []

    for rnd in trange(1, rounds + 1, desc=args.method):
        round_start = time.perf_counter()
        selected = schedule[rnd - 1]

        stage1_global_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        results = [
            train_client(
                cid,
                model,
                train_set,
                client_indices[cid],
                cfg,
                device,
                method="fedavg",
                compute_info_signals=True,
                round_idx=rnd,
            )
            for cid in selected
        ]

        if is_oracle:
            for r in results:
                r.mi_proxy = -0.8 if r.client_id in corrupted_clients else 0.0
                r.entropy = 0.0
                r.update_norm = 0.0
                r.train_loss_std = 0.0

        if is_oracle and controller is not None:
            weights, info = controller.compute_weights(results, round_idx=rnd)
        else:
            ns = np.array([r.num_samples for r in results], dtype=float)
            weights = (ns / ns.sum()).tolist()
            info = baseline_signal_info(results, weights)

        local_states = [r.state for r in results]
        new_state = average_states(local_states, weights)
        model.load_state_dict(new_state, strict=True)

        # Per-round diagnostics row.
        try:
            diag = build_round_diag(
                rnd,
                controller_method,
                None,
                results,
                weights,
                _sanitize_for_json(diag_export_info(controller_method, info)),
                stage1_global_state,
                new_state,
                average_states,
                param_keys,
            )
            with open(run_dir / "diagnostics.jsonl", "a", encoding="utf-8") as df:
                df.write(json.dumps(diag, allow_nan=False) + "\n")
        except ValueError as exc:
            raise RuntimeError(f"diagnostics build failed at round {rnd}: {exc}")

        upload_mb += model_size_mb * len(selected)
        download_mb += model_size_mb * len(selected)
        if is_oracle:
            metadata_mb += model_size_mb * 0.01 * len(selected)
        cumulative_comm_mb = upload_mb + download_mb + metadata_mb

        eval_every = cfg["evaluation"].get("eval_every", 1)
        if rnd % eval_every == 0:
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

            logger.log(
                {
                    "round": rnd,
                    "method": controller_method,
                    "test_acc": acc,
                    "test_loss": metrics["loss"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1": metrics["macro_f1"],
                    "train_loss": train_loss,
                    "train_loss_std": train_loss_std,
                    "mean_mi": info.get("mean_mi", 0.0),
                    "mean_entropy": info.get("mean_entropy", 0.0),
                    "mean_js": info.get("mean_js", 0.0),
                    "mean_stability": info.get("mean_stability", 0.0),
                    "mean_score": info.get("mean_score", 0.0),
                    "weight_var": info.get("weight_var", 0.0),
                    "temperature_used": info.get(
                        "temperature_used",
                        float(cfg["infoopt"].get("temperature", 1.0)),
                    ),
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
                    "client_scores": ";".join(
                        f"{s:.4f}" for s in info.get("scores", [])
                    ),
                    "nu": info.get("nu", 0.0),
                    "epsilon": info.get("epsilon", 0.0),
                    "mixture_js": info.get("mixture_js", 0.0),
                    "fallback_used": int(info.get("fallback_used", False)),
                    "fallback_reason": info.get("fallback_reason", "none"),
                    # RA-InfoOpt compatibility fields.
                    "heterogeneity_raw": 0.0,
                    "heterogeneity_normalized": 0.0,
                    "reliability_separation": 0.0,
                    "reliability_rank": 0.0,
                    "reliability_temporal": 0.0,
                    "reliability_total": 0.0,
                    "alpha_raw": 0.0,
                    "alpha_ema": 0.0,
                    "alpha_safe": 0.0,
                    "js_fedavg": 0.0,
                    "js_infoopt": 0.0,
                    "js_final": 0.0,
                    "enc_fedavg": 0.0,
                    "enc_infoopt": 0.0,
                    "enc_final": 0.0,
                    "weight_l1_infoopt_vs_fedavg": 0.0,
                    "weight_l1_final_vs_fedavg": 0.0,
                    "infoopt_score_std": 0.0,
                    "robust_separation_raw": 0.0,
                    "overlap_count": 0,
                    "rank_rho": float("nan"),
                    "rank_mapped": float("nan"),
                    "rank_valid": 0,
                    "score_persistence": float("nan"),
                    "score_persistence_valid": 0,
                    "blend_fraction": float("nan"),
                    "blend_identity_error": 0.0,
                    "blend_identity_error_abs": 0.0,
                    "js_deviation_limit": 0.0,
                    "enc_deviation_limit": 0.0,
                    "js_fedavg_deviation": 0.0,
                    "js_infoopt_deviation": 0.0,
                    "js_blend_deviation": 0.0,
                    "fedavg_feasible": 1,
                    "infoopt_feasible": 0,
                    "blend_feasible": 0,
                }
            )
            print(
                f"round={rnd} method={args.method} acc={acc:.4f} "
                f"f1={metrics['macro_f1']:.4f} loss={metrics['loss']:.4f} "
                f"comm_mb={cumulative_comm_mb:.2f} best_acc={best_acc:.4f}",
                flush=True,
            )
            per_class_f1_log.append(
                {"round": rnd, "per_class_f1": metrics["per_class_f1"]}
            )

            if (
                is_oracle
                and controller is not None
                and controller.should_stop(rnd, acc, info)
            ):
                print(f"Early stopping at round {rnd}")
                early_stop_triggered = True
                break

    logger.close()

    final_metrics = evaluate(model, test_loader, device)
    summary = {
        "method": args.method,
        "controller_method": controller_method,
        "seed": seed,
        "alpha": alpha,
        "rounds_completed": rnd,
        "early_stop_triggered": early_stop_triggered,
        "best_acc": best_acc,
        "best_round": best_round,
        "rounds_to_90": rounds_to_90,
        "rounds_to_95": rounds_to_95,
        "final_acc": final_metrics["acc"],
        "final_loss": final_metrics["loss"],
        "final_macro_f1": final_metrics["macro_f1"],
        "corrupted_clients": corrupted_clients,
        "run_dir": str(run_dir),
    }
    save_json(summary, str(run_dir / "summary.json"))
    if per_class_f1_log:
        save_json(per_class_f1_log, str(run_dir / "per_class_f1_per_round.json"))
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
