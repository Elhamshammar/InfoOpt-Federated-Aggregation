#!/usr/bin/env python3
"""Stage-4 technical smoke: ResNet18SmallImage GPU memory and timing in
a DISPOSABLE namespace. Never run by the launcher; invoked explicitly.

Covers the REAL per-dataset input dimensions and BOTH configured batch
sizes:
  - training path: forward+backward+optimizer step at batch 64;
  - probe/evaluation path: eval-mode forward (incl. the
    return_features=True signal interface) at batch 256.
Datasets: cifar100 3x32x32 (100 classes), eurosat 3x64x64 (10),
pathmnist 3x28x28 (9). No resizing anywhere (adaptive pooling).

Timing is synchronized (torch.cuda.synchronize before/after on CUDA).
Peak memory is recorded per path. GPU identity is recorded.

Server-readiness mode: --require-cuda aborts unless a CUDA device is
present. Output JSON is written with mode "x" (no overwrite).

Usage (server, GPU 1):
  CUDA_VISIBLE_DEVICES=1 python smoke_resnet_runtime.py \
      --out-dir ./datasets \
      --require-cuda

If any configured batch does not fit, this reports the failure rather
than changing anything; report the issue before proposing an amendment
(no silent batch-size change is permitted).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshot"))

import torch

DATASETS = {  # name -> (num_classes, H, W)
    "cifar100": (100, 32, 32),
    "eurosat": (10, 64, 64),
    "pathmnist": (9, 28, 28),
}
TRAIN_BATCH = 64
EVAL_BATCH = 256


def _timed_train_steps(model, x, y, steps, device):
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device == "cuda" else None)
    assert torch.isfinite(loss.detach()).item()
    return dt, peak


def _timed_eval(model, x, device):
    model.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, feat = model(x, return_features=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device == "cuda" else None)
    assert logits.shape == (x.shape[0], model.classifier.out_features)
    assert feat.shape == (x.shape[0], model.classifier.in_features)
    assert bool(torch.isfinite(logits).all())
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--require-cuda", action="store_true",
                    help="server-readiness mode: abort unless CUDA "
                         "is available")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_resnet_runtime.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite: {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise SystemExit("SMOKE ABORT: --require-cuda set but no CUDA "
                         "device is available")

    from models.build import build_model
    n_params = sum(p.numel() for p in build_model(
        {"model": {"name": "ResNet18SmallImage", "use_aib": False},
         "dataset": {"num_classes": 100, "input_channels": 3}}
    ).parameters())
    results = {}
    oom = []
    for ds, (nc, h, w) in DATASETS.items():
        cfg = {"model": {"name": "ResNet18SmallImage", "use_aib": False},
               "dataset": {"num_classes": nc, "input_channels": 3}}
        try:
            model = build_model(cfg).to(device)
            xt = torch.randn(TRAIN_BATCH, 3, h, w, device=device)
            yt = torch.randint(0, nc, (TRAIN_BATCH,), device=device)
            tr_dt, tr_peak = _timed_train_steps(model, xt, yt,
                                                args.steps, device)
            xe = torch.randn(EVAL_BATCH, 3, h, w, device=device)
            ev_dt, ev_peak = _timed_eval(model, xe, device)
            results[ds] = {
                "input": [3, h, w], "num_classes": nc,
                "train_batch": TRAIN_BATCH,
                "train_sec_per_step": tr_dt,
                "train_peak_gpu_mb": tr_peak,
                "eval_batch": EVAL_BATCH,
                "eval_sec": ev_dt,
                "eval_peak_gpu_mb": ev_peak,
            }
            del model, xt, yt, xe
            if device == "cuda":
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            oom.append(ds)
            if device == "cuda":
                torch.cuda.empty_cache()

    rec = {
        "backbone": "ResNet18SmallImage",
        "device": device,
        "gpu_identity": (torch.cuda.get_device_name(0)
                         if device == "cuda" else None),
        "torch": torch.__version__,
        "steps": args.steps,
        "parameters": n_params,
        "results": results,
        "oom_failures": oom,
        "note": "disposable-namespace technical smoke; not a training "
                "run; measured on THIS device only; must not be "
                "presented as SmallCNN costs or as server measurements "
                "when run elsewhere",
    }
    out_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    if oom:
        raise SystemExit(f"SMOKE OOM at configured batch sizes: {oom}. "
                         f"Report before proposing any amendment.")


if __name__ == "__main__":
    main()
