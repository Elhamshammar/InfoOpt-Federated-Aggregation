#!/usr/bin/env python3
"""Post-run integrity verifier for the four-arm experiment (arm-aware,
hardened for malformed evidence).

Every parse/load is guarded: malformed config/CSV/checkpoint produces a
STRUCTURED failure entry and a nonzero exit — never a traceback.
terminal_acc stays null unless verification passes. The report JSON is
written with allow_nan=False.

Checks:
  1. all evidence files exist;
  2. config_used.json matches the expected ARM (dataset, seed, method,
     effective use_aib, common training settings, early stop disabled,
     probe disabled);
  3. metrics.csv: exact rounds 1..100 unique; test_acc finite and in
     [0,1]; test_loss finite; no early stop triggered;
  4. recorded cohort equals the frozen schedule;
  5. partition_used.json equals the frozen shared partition;
  6. final_model.pt: loads (weights_only), all floating tensors finite,
     strict load into the recovered model for the arm's AIB setting, AIB
     parameter presence matches the arm;
  7. summary.json: early_stop_triggered is false;
  8. exporter-key selftest.
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
ARM_METHOD = {"fedavg_aiboff": ("fedavg", False),
              "fedavg_aibon": ("fedavg", True),
              "infoopt_aiboff": ("InfoOpt-Full", False),
              "infoopt_aibon": ("InfoOpt-Full", True)}
DS_NAME = {"cifar100": "CIFAR100", "eurosat": "EuroSAT",
           "pathmnist": "PathMNIST"}
EVIDENCE = ("config_used.json", "metrics.csv", "summary.json",
            "final_model.pt", "partition_used.json")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model_ctor(snap, use_aib):
    import importlib.util
    import sys as _sys
    import types
    pkg = types.ModuleType("snap_models")
    pkg.__path__ = []
    _sys.modules["snap_models"] = pkg
    for mod in ("aib", "cnn"):
        spec = importlib.util.spec_from_file_location(
            f"snap_models.{mod}", snap / "models" / f"{mod}.py")
        m = importlib.util.module_from_spec(spec)
        _sys.modules[f"snap_models.{mod}"] = m
        spec.loader.exec_module(m)
    return _sys.modules["snap_models.cnn"].SmallCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=list(ARM_METHOD))
    ap.add_argument("--schedules-dir", required=True)
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    failures = []
    report = {"run_dir": str(rd), "dataset": args.dataset,
              "seed": args.seed, "arm": args.arm, "failures": failures,
              "terminal_acc": None}

    for fn in EVIDENCE:
        if not (rd / fn).is_file():
            failures.append(f"missing {fn}")

    exp_method, exp_aib = ARM_METHOD[args.arm]
    if (rd / "config_used.json").is_file():
        try:
            c = json.load(open(rd / "config_used.json", encoding="utf-8"))
            cfg = c.get("config", c)
            if cfg.get("dataset", {}).get("name") != DS_NAME[args.dataset]:
                failures.append("dataset mismatch in config")
            if int(cfg.get("seed", -1)) != args.seed:
                failures.append("seed mismatch in config")
            if c.get("method") != exp_method:
                failures.append(f"method {c.get('method')!r} != "
                                f"{exp_method!r}")
            if cfg.get("model", {}).get("use_aib") is not exp_aib:
                failures.append(f"effective use_aib "
                                f"{cfg['model'].get('use_aib')} != {exp_aib}")
            tr = cfg.get("training", {})
            for k, v in COMMON.items():
                if tr.get(k) != v:
                    failures.append(f"training.{k}={tr.get(k)} != {v}")
            part = cfg.get("partition", {})
            if part.get("alpha") != 0.3 or part.get("num_clients") != 10:
                failures.append("partition settings mismatch")
            if cfg.get("infoopt", {}).get("early_stop", {}).get(
                    "enabled") not in (False, None):
                failures.append("early stopping not disabled")
            if cfg.get("counterfactual_probe", {}).get("enabled", False):
                failures.append("counterfactual probe not disabled")
        except Exception as e:
            failures.append(f"malformed config_used.json: {e}")

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
                    failures.append(f"malformed round value: "
                                    f"{r.get('round')!r}")
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
                        failures.append(f"test_acc out of [0,1] at round "
                                        f"{r.get('round')}")
                except (KeyError, TypeError, ValueError):
                    failures.append(f"malformed test_acc at round "
                                    f"{r.get('round')}")
                try:
                    lv = float(r["test_loss"])
                    if not math.isfinite(lv):
                        failures.append(f"non-finite test_loss round "
                                        f"{r.get('round')}")
                except (KeyError, TypeError, ValueError):
                    failures.append(f"malformed test_loss at round "
                                    f"{r.get('round')}")
            if any(r.get("early_stop_triggered") == "True" for r in rows):
                failures.append("early_stop_triggered is true")
        except Exception as e:
            failures.append(f"malformed metrics.csv: {e}")

        sched_path = (Path(args.schedules_dir)
                      / f"cohort_schedule_{args.dataset}_seed{args.seed}.json")
        if not sched_path.is_file():
            failures.append(f"frozen schedule missing: {sched_path}")
        elif rows:
            try:
                sched = json.load(open(sched_path,
                                       encoding="utf-8"))["schedule"]
                actual = [[int(x) for x in r["selected_clients"].split(";")]
                          for r in rows]
                if actual != sched:
                    mism = [i + 1 for i, (a, b)
                            in enumerate(zip(actual, sched)) if a != b]
                    failures.append(f"cohort mismatch at rounds {mism[:5]}")
            except Exception as e:
                failures.append(f"schedule comparison failed: {e}")

    if (rd / "partition_used.json").is_file():
        frozen = (Path(args.schedules_dir)
                  / f"partition_{args.dataset}_seed{args.seed}.json")
        if not frozen.is_file():
            failures.append(f"frozen partition missing: {frozen}")
        else:
            try:
                used = json.load(open(rd / "partition_used.json",
                                      encoding="utf-8"))
                want = json.load(open(frozen, encoding="utf-8"))
                if used != want["client_indices"]:
                    failures.append("partition_used != frozen shared "
                                    "partition")
            except Exception as e:
                failures.append(f"malformed partition evidence: {e}")

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
            has_aib = any(k.endswith("conv1.3.gamma") for k in sd)
            if has_aib != exp_aib:
                failures.append(f"checkpoint AIB presence {has_aib} != "
                                f"expected {exp_aib}")
            snap = Path(__file__).resolve().parent / "snapshot"
            try:
                SmallCNN = _load_model_ctor(snap, exp_aib)
                model = SmallCNN(
                    num_classes=(10 if args.dataset == "eurosat"
                                 else 9 if args.dataset == "pathmnist"
                                 else 100),
                    input_channels=3, use_aib=exp_aib)
                model.load_state_dict(sd, strict=True)
            except Exception as e:
                failures.append(f"checkpoint does not load strictly into "
                                f"the expected model: {str(e)[:200]}")
        except Exception as e:
            failures.append(f"malformed checkpoint: {e}")

    if (rd / "summary.json").is_file():
        try:
            s = json.load(open(rd / "summary.json", encoding="utf-8"))
            if s.get("early_stop_triggered") not in (False, None):
                failures.append("summary early_stop_triggered is true")
        except Exception as e:
            failures.append(f"malformed summary.json: {e}")

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
            a = float(rows[-1]["test_acc"])
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
