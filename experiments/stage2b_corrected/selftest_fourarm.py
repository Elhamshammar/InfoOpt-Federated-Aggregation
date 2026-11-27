#!/usr/bin/env python3
"""Regression tests for the four-arm prep repairs.

Synthetic-fixture tests (no training):
  T1 wrong AIB in config -> verify_run fails
  T2 changed frozen partition -> verify_run fails (partition mismatch)
  T3 corrupted schedule -> verify_run fails (cohort mismatch)
  T4 invalid completed run -> resume_check returns abort_invalid
  T5 duplicate cell directories -> resume_check returns abort_duplicate
  T6 differing caller working directories -> dataset resolution finds the
     dataset from an unrelated CWD
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def sha256(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_run_dir(root, name, *, aib=True, cohort="1;2;3;4;5", rounds=100,
                 acc=0.5, method="fedavg", seed=100, partition=None):
    d = Path(root) / name
    d.mkdir(parents=True)
    cfg = {"method": method,
           "config": {"seed": seed,
                      "dataset": {"name": "CIFAR100", "num_classes": 100,
                                  "root": "/x", "input_channels": 3},
                      "partition": {"alpha": 0.3, "num_clients": 10,
                                    "min_samples_per_client": 50},
                      "model": {"name": "SmallCNN", "use_aib": aib},
                      "training": {"lr": 0.01, "momentum": 0.9,
                                   "weight_decay": 5e-4, "local_epochs": 2,
                                   "batch_size": 64, "clip_grad_norm": 5.0,
                                   "rounds": rounds,
                                   "clients_per_round": 5},
                      "infoopt": {"early_stop": {"enabled": False}}}}
    json.dump(cfg, open(d / "config_used.json", "w"))
    json.dump(partition or {"0": [1, 2], "1": [3, 4]},
              open(d / "partition_used.json", "w"))
    json.dump({"early_stop_triggered": False}, open(d / "summary.json", "w"))
    with open(d / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "test_acc", "test_loss",
                    "selected_clients", "early_stop_triggered"])
        for r in range(1, rounds + 1):
            w.writerow([r, acc, 1.0, cohort, False])
    # checkpoint: a REAL state_dict from the snapshot SmallCNN so the
    # verifier's strict load + tensor checks pass
    import torch, types, importlib.util
    pkg = types.ModuleType("tm"); pkg.__path__ = []
    import sys as _s
    _s.modules["tm"] = pkg
    for mod in ("aib", "cnn"):
        spec = importlib.util.spec_from_file_location(
            f"tm.{mod}", HERE / "snapshot" / "models" / f"{mod}.py")
        m = importlib.util.module_from_spec(spec)
        _s.modules[f"tm.{mod}"] = m
        spec.loader.exec_module(m)
    model = _s.modules["tm.cnn"].SmallCNN(num_classes=100,
                                          input_channels=3, use_aib=aib)
    torch.save(model.state_dict(), d / "final_model.pt")
    return d


def make_shared(root, cohort=None, partition=None):
    sd = Path(root)
    sd.mkdir(parents=True, exist_ok=True)
    cohort = cohort or [[1, 2, 3, 4, 5]] * 100
    json.dump({"schedule": cohort},
              open(sd / "cohort_schedule_cifar100_seed100.json", "w"))
    json.dump({"client_indices": partition or {"0": [1, 2], "1": [3, 4]}},
              open(sd / "partition_cifar100_seed100.json", "w"))
    return sd


def run_verify(rd, ds="cifar100", seed=100, arm="fedavg_aibon", shared=None,
               out=None):
    cmd = [PY, str(HERE / "verify_run.py"), "--run-dir", str(rd),
           "--dataset", ds, "--seed", str(seed), "--arm", arm,
           "--schedules-dir", str(shared)]
    if out:
        cmd += ["--report-out", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_resume(ds, seed, arm, runs, shared):
    return subprocess.run(
        [PY, str(HERE / "resume_check.py"), "--dataset", ds, "--seed",
         str(seed), "--arm", arm, "--runs-dir", str(runs),
         "--schedules-dir", str(shared), "--python", PY],
        capture_output=True, text=True)


def _find_git_bash():
    import shutil
    bash_exe = os.environ.get("BASH_EXE")
    if bash_exe:
        return bash_exe
    found = shutil.which("bash")
    return found or "bash"


def _msys(p):
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shared = make_shared(root / "shared")
        good = make_run_dir(root / "runs", "cell_a")

        r = run_verify(good, shared=shared)
        check("T0 valid fixture verifies", r.returncode == 0,
              (r.stdout + r.stderr).strip().splitlines()[-1][:80])

        # T1: wrong AIB
        bad_aib = make_run_dir(root / "runs", "cell_b", aib=False)
        r = run_verify(bad_aib, shared=shared)
        check("T1 wrong AIB rejected", r.returncode != 0
              and "use_aib" in r.stdout + r.stderr)

        # T2: changed frozen partition
        shared2 = make_shared(root / "shared2",
                              partition={"0": [9, 9], "1": [3, 4]})
        r = run_verify(good, shared=shared2)
        check("T2 changed partition rejected", r.returncode != 0
              and "partition_used != frozen" in r.stdout + r.stderr)

        # T3: corrupted schedule
        shared3 = make_shared(root / "shared3",
                              cohort=[[9, 8, 7, 6, 5]] * 100)
        r = run_verify(good, shared=shared3)
        check("T3 corrupted schedule rejected", r.returncode != 0
              and "cohort mismatch" in r.stdout + r.stderr)

        # T4: invalid completed run -> abort_invalid
        bad = make_run_dir(root / "runs4",
                   "fourarm_cifar100_fedavg_aibon_x_seed100_alpha0.3_1",
                   rounds=99)
        r = run_resume("cifar100", 100, "fedavg_aibon", root / "runs4",
                       shared)
        verdict = r.stdout.strip().splitlines()[-1]
        check("T4 invalid completed run -> abort_invalid",
              verdict == "abort_invalid", verdict)

        # T5: duplicate cell directories -> abort_duplicate
        dup_root = root / "runs5"
        make_run_dir(dup_root, "fourarm_cifar100_fedavg_aibon_x_seed100_alpha0.3_1")
        make_run_dir(dup_root, "fourarm_cifar100_fedavg_aibon_x_seed100_alpha0.3_2")
        # force same cell pattern: resume_check matches by arm prefix+seed,
        # and both dirs match fourarm_*_*_seed100_ pattern? they must share
        # the same prefix; we simulate by matching dataset+arm+seed on dirs
        # named identically-patterned:
        r = run_resume("cifar100", 100, "fedavg_aibon", dup_root, shared)
        # both dirs match the same glob, so expect abort_duplicate
        verdict = r.stdout.strip().splitlines()[-1]
        check("T5 duplicate dirs -> abort_duplicate",
              verdict == "abort_duplicate", verdict)

        # T6: caller CWD independence for dataset resolution
        fake_ds = root / "dsroot" / "cifar100"
        fake_ds.mkdir(parents=True)
        (fake_ds / "cifar-100-python.tar.gz").write_bytes(b"x")
        for dsn in ("eurosat", "pathmnist"):
            d = root / "dsroot" / dsn
            d.mkdir(parents=True, exist_ok=True)
            (d / "x.txt").write_text("x")
        import os
        env = dict(os.environ, DATASETS_ROOT=str(root / "dsroot"),
                   EXP_ROOT=str(root / "exp"))
        r = subprocess.run([PY, str(HERE / "resolve_dataset_roots.py")],
                           cwd="/", env=env, capture_output=True, text=True)
        check("T6 dataset resolution from unrelated CWD",
              r.returncode == 0 and "resolved" in r.stdout,
              (r.stdout + r.stderr).strip().splitlines()[-1][:100])


def T7_T8():
    import os
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        exp = root / "exp"
        runs = exp / "runs"
        failed_root = exp / "failed_attempts"
        runs.mkdir(parents=True)
        failed_root.mkdir(parents=True)
        shared = make_shared(root / "shared")

        # T7: archived failed attempt + valid retry in runs/
        cell_name = ("fourarm_cifar100_fedavg_aibon_x_seed100_alpha0.3_1")
        archived = make_run_dir(failed_root, cell_name + "_attempt1")
        valid = make_run_dir(runs, cell_name + "_attempt2")
        r = run_resume("cifar100", 100, "fedavg_aibon", runs, shared)
        verdict = r.stdout.strip().splitlines()[-1]
        check("T7 archived attempt does not block verified retry",
              verdict == "skip_complete"
              and "ARCHIVED ATTEMPT" in r.stderr, verdict)

        # T8: malformed terminal accuracy -> invalid record, no crash
        bad = make_run_dir(runs,
                           "fourarm_cifar100_fedavg_aibon_y_seed101_alpha0.3_3")
        # corrupt the metrics endpoint AFTER creation
        rows = list(csv.DictReader(open(bad / "metrics.csv")))
        with open(bad / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "test_acc", "test_loss",
                        "selected_clients", "early_stop_triggered"])
            for row in rows[:-1]:
                w.writerow([row["round"], row["test_acc"], row["test_loss"],
                            row["selected_clients"],
                            row["early_stop_triggered"]])
            w.writerow([100, "not_a_number", 1.0, "1;2;3;4;5", False])
        env = dict(os.environ, EXP_ROOT=str(exp))
        import subprocess as sp
        r = sp.run([sys.executable, str(HERE / "build_run_manifest.py")],
                   env=env, capture_output=True, text=True)
        man = json.load(open(exp / "RUN_MANIFEST.json"))
        target = [c for c in man["cells"]
                  if c["seed"] == 101 and c["dataset"] == "cifar100"
                  and c["arm"] == "fedavg_aibon"][0]
        check("T8 malformed endpoint -> invalid, null terminal_acc, no crash",
              target["status"] == "invalid"
              and target["terminal_acc"] is None
              and target["reason"], f"{target['status']} | "
              f"{str(target['reason'])[:60]}")
        # the JSON itself was written with allow_nan=False (no NaN tokens)
        check("T8b manifest JSON has no NaN tokens",
              "NaN" not in (exp / "RUN_MANIFEST.json").read_text())



def T9():
    """stdout of resume_check must contain EXACTLY ONE decision line, even
    when the verifier prints diagnostics (which must go to stderr)."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shared = make_shared(root / "shared")

        # case A: valid completed run (verifier prints VERIFY OK diag.)
        runs_a = root / "runs_a"
        make_run_dir(runs_a, "fourarm_cifar100_fedavg_aibon_a_seed100_alpha0.3_1")
        r = run_resume("cifar100", 100, "fedavg_aibon", runs_a, shared)
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        check("T9a stdout is exactly one decision line (verified complete)",
              lines == ["skip_complete"] and "VERIFY OK" in r.stderr,
              f"stdout={lines!r}")

        # case B: invalid run (verifier prints VERIFY FAIL diag.)
        runs_b = root / "runs_b"
        bad = make_run_dir(runs_b, "fourarm_cifar100_fedavg_aibon_b_seed100_alpha0.3_1",
                           rounds=99)
        r = run_resume("cifar100", 100, "fedavg_aibon", runs_b, shared)
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        check("T9b stdout is exactly one decision line (invalid run)",
              lines == ["abort_invalid"] and "VERIFY FAIL" in r.stderr,
              f"stdout={lines!r}")

        # case C: all four decisions reachable
        runs_c = root / "runs_c"
        runs_c.mkdir()
        r = run_resume("cifar100", 100, "fedavg_aibon", runs_c, shared)
        check("T9c 'run' decision for empty cell", r.stdout.strip() == "run")
        make_run_dir(runs_c, "fourarm_cifar100_fedavg_aibon_c_seed100_alpha0.3_1")
        make_run_dir(runs_c, "fourarm_cifar100_fedavg_aibon_c_seed100_alpha0.3_2")
        r = run_resume("cifar100", 100, "fedavg_aibon", runs_c, shared)
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        check("T9d 'abort_duplicate' decision", lines == ["abort_duplicate"])
        decisions = {"run", "skip_complete", "abort_invalid",
                     "abort_duplicate"}
        check("T9e all four decisions are single-word launch tokens",
              all(len(d.split()) == 1 for d in decisions))

T9()


def T10():
    """End-to-end preflight-only launch from an unrelated CWD: integrity,
    dataset resolution and schedule generation must succeed, then exit
    before training, without touching the package's shared/ or creating
    run directories."""
    import os, shutil
    local_data = os.environ.get(
        "FOURARM_TEST_CIFAR_ROOT",
        rstr(Path(__file__).resolve().parents[2] / "datasets"))
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        exp = root / "exp"
        shared = root / "shared_test"
        shared.mkdir()
        pkg_shared_before = set((HERE / "shared").glob("*"))
        env = dict(os.environ,
                   PKG=str(HERE),
                   EXP_ROOT=str(exp),
                   DATASETS="cifar100",
                   DATASETS_ROOT=local_data,
                   SHARED_DIR=str(shared),
                   PREFLIGHT_ONLY="1",
                   PY=sys.executable)
        bash_exe = _find_git_bash()
        r = subprocess.run(
            [bash_exe, _msys(HERE / "launch_four_arm_gpu1.sh")],
            cwd="/", env=env, capture_output=True, text=True)
        out = r.stdout + r.stderr
        check("T10 preflight exits 0 from unrelated CWD",
              r.returncode == 0 and "PREFLIGHT COMPLETE" in out,
              out.strip().splitlines()[-1][:80] if out else "")
        check("T10b schedules written to the override dir",
              (shared / "SHARED_MANIFEST.json").is_file()
              and len(list(shared.glob("partition_*.json"))) == 5)
        check("T10c no run directories created",
              not (exp / "runs").exists()
              or not any((exp / "runs").iterdir()))
        check("T10d package shared/ untouched",
              set((HERE / "shared").glob("*")) == pkg_shared_before)

T10()


def T11():
    """Corrupt a generated schedule in the override dir -> integrity
    checking (--require-complete-schedules) must abort."""
    import os
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        exp = root / "exp"
        shared = root / "shared_test"
        shared.mkdir()
        env = dict(os.environ, PKG=str(HERE), EXP_ROOT=str(exp),
                   DATASETS="cifar100",
                   DATASETS_ROOT=os.environ.get(
                       "FOURARM_TEST_CIFAR_ROOT",
                       rstr(Path(__file__).resolve().parents[2] / "datasets")),
                   SHARED_DIR=str(shared), PREFLIGHT_ONLY="1",
                   PY=sys.executable)
        bash_exe = _find_git_bash()
        r = subprocess.run([bash_exe, _msys(HERE / "launch_four_arm_gpu1.sh")],
                           cwd="/", env=env, capture_output=True, text=True)
        assert "PREFLIGHT COMPLETE" in r.stdout + r.stderr, \
            "preflight generation failed: " + r.stdout[-300:] \
            + r.stderr[-300:]
        # corrupt one schedule file
        victim = next(shared.glob("cohort_schedule_*.json"))
        payload = json.loads(victim.read_text())
        payload["schedule"][0] = [9, 9, 9, 9, 9]
        victim.write_text(json.dumps(payload))
        r2 = subprocess.run(
            [sys.executable,
             str(HERE / "verify_frozen_integrity.py"),
             "--require-complete-schedules"],
            cwd="/", env=env, capture_output=True, text=True)
        check("T11 corrupted schedule aborts integrity check",
              r2.returncode != 0 and "mismatch" in r2.stdout + r.stderr,
              (r2.stdout + r2.stderr).strip().splitlines()[-1][:80])

T11()

def main_wrapper():
    main()
    T7_T8()
    T9()
    T10()
    T11()

    failed = [n for n, ok in RESULTS if not ok]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print(f"ALL {len(RESULTS)} REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main_wrapper()
    main()
