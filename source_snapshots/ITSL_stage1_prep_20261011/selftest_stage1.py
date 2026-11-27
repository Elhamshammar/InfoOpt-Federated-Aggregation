#!/usr/bin/env python3
"""Stage-1 regression suite (CPU, meaningful synthetic fixtures).

Strengthened version — every claimed guarantee is exercised against the
production code path, not a local re-derivation:

  S1 aggregation/damping via the PRODUCTION functions
     (utils.aggregation.average_states / apply_magnitude_damping):
     uniform and FedAvg weightings, BOTH damping coefficients (0.5, 0.75),
     floating buffers damped, integer buffer = first selected client's value.
  S2 shuffle: score multiset preserved, deterministic permutation, logged;
     training RNG (python, numpy, torch) not advanced; loader draws with
     num_workers=2 (the configured setting) identical before/after.
  S3 solver: closed-form path taken when feasible; a FORCED constrained
     case that must invoke SLSQP successfully (feasibility + status
     asserted); forced failure -> fallback flagged; non-finite raises;
     zero-norm diagnostics -> null with reason.
  S4 diagnostics builder rejects non-finite updates.
  S5 launcher preflight subprocess from an unrelated CWD — portable bash
     detection (Git Bash on Windows, system bash elsewhere) and an
     explicit test dataset root (STAGE1_TEST_DATA_ROOT); reported as SKIP,
     not PASS, when no dataset root is available.
  S6 initialization-hash identity (unchanged).
  S7 probe context: enabled-vs-disabled comparison from identical starting
     states (trained state AND full python/numpy/torch RNG states AND
     subsequent draws must be identical); different arm contexts must give
     identical derived probe seeds, identical probe draws and identical
     signals, exercised through a stochastic dataset transform.
  S8 client alignment: per-client EMA stores across multiple rounds match
     hand-computed values; histogram rows reaching the solver stay in
     original client order while only the scores are permuted.
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshot"))

RESULTS = []
SKIPPED = []
REQUIRE_PREFLIGHT = "--require-preflight" in sys.argv


def check(name, cond, detail=""):
    RESULTS.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def skip(name, reason):
    """Record a check as NOT EXECUTED. Skips are reported separately and
    never counted as passed. With --require-preflight (launch readiness),
    a skipped preflight check is a failure."""
    if REQUIRE_PREFLIGHT:
        RESULTS.append((name, False))
        print(f"  [FAIL] {name} SKIPPED under --require-preflight: {reason}")
    else:
        SKIPPED.append((name, reason))
        print(f"  [SKIP] {name} {reason}")


def _np_state_eq(a, b):
    return (a[0] == b[0] and np.array_equal(a[1], b[1])
            and a[2] == b[2] and a[3] == b[3] and a[4] == b[4])


class FakeResult:
    def __init__(self, cid, n, signals, state):
        self.client_id = cid
        self.num_samples = n
        self.mi_proxy, self.entropy, self.update_norm, self.train_loss_std = \
            signals
        self.label_hist = np.eye(10)[cid % 10].astype(float).copy()
        self.state = state
        self.score = 0.0


def make_results(C=5, aib=False):
    out = []
    for i in range(C):
        st = {"conv.weight": torch.ones(2, 2) * i,
              "bn.running_mean": torch.ones(2) * i,
              "bn.num_batches_tracked": torch.tensor(i)}
        out.append(FakeResult(i, 100 + 10 * i,
                              (0.1 * i, 0.2 * i, 0.05 * i, 0.01 * i), st))
    return out


class StochasticImages(Dataset):
    """Dataset whose transform draws fresh torch RNG noise on every access,
    so probe/training randomness is genuinely exercised. Module-level so it
    is picklable for DataLoader workers."""

    def __init__(self, n=64, shape=(3, 32, 32), num_classes=10, gen_seed=123):
        g = torch.Generator().manual_seed(gen_seed)
        self.x = torch.randn(n, *shape, generator=g)
        self.targets = torch.randint(0, num_classes, (n,),
                                     generator=g).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i] + 0.05 * torch.randn_like(self.x[i]), \
            int(self.targets[i])


def _find_bash():
    bash_exe = os.environ.get("BASH_EXE")
    if bash_exe:
        return bash_exe
    return shutil.which("bash")


def _msys(p):
    s_ = str(p).replace("\\", "/")
    return ("/" + s_[0].lower() + s_[2:]) if len(s_) > 1 and \
        s_[1] == ":" else s_


def main():
    from methods.infoopt import (InfoOptController, solve_entropic_weights,
                                 _js_and_grad, stage1_shuffle_seed)
    from utils.stage1_diagnostics import build_round_diag
    from utils.aggregation import (average_states, apply_magnitude_damping)

    # ---- S1: PRODUCTION weights + damping code ----
    # states with a parameter, a floating buffer and an integer buffer;
    # first client carries a distinctive integer-buffer value.
    states = [
        {"w": torch.tensor([1.0 * i, -0.5 * i, 0.25 * i]),
         "var": torch.full((3,), float(i) + 0.5),
         "nbt": torch.tensor(100 + 7 * i)}
        for i in range(5)
    ]
    g0 = {"w": torch.zeros(3), "var": torch.ones(3),
          "nbt": torch.tensor(5)}
    C = len(states)

    # S1a: uniform arm — train.py branch is [1.0/len(results)] * len(results)
    uniform_w = [1.0 / C] * C
    out_u = average_states(states, uniform_w)
    mean_w = sum(s["w"] for s in states) / C
    mean_var = sum(s["var"] for s in states) / C
    check("S1a production uniform aggregate equals arithmetic mean",
          abs(sum(uniform_w) - 1.0) < 1e-15
          and torch.allclose(out_u["w"], mean_w)
          and torch.allclose(out_u["var"], mean_var))

    # S1b: FedAvg arm — train.py branch is (ns / ns.sum()).tolist()
    ns = np.array([100, 200, 300, 400, 500], dtype=float)
    fa_w = (ns / ns.sum()).tolist()
    out_f = average_states(states, fa_w)
    exp_w = sum(w * s["w"] for w, s in zip(fa_w, states))
    check("S1b production FedAvg aggregate equals n_i/sum(n_i) weighting",
          torch.allclose(out_f["w"], exp_w, atol=1e-7)
          and abs(fa_w[0] - 100 / 1500) < 1e-15)

    # S1c: BOTH damping coefficients through production damping, applied to
    # the production aggregate; every floating entry (param AND buffer)
    # moves by exactly gamma of the displacement.
    ok_c = True
    for gamma in (0.5, 0.75):
        damped = apply_magnitude_damping(out_f, g0, gamma)
        for k in ("w", "var"):
            ok_c &= torch.allclose(
                damped[k], g0[k] + gamma * (out_f[k] - g0[k]), atol=1e-7)
    check("S1c production damping exact for gamma in {0.5, 0.75} "
          "(params and floating buffers)", ok_c)

    # S1d: integer buffer comes from the FIRST selected client's state,
    # regardless of weights, and damping leaves it untouched.
    check("S1d integer buffer takes first selected client's value",
          out_f["nbt"].item() == 100
          and out_u["nbt"].item() == 100
          and apply_magnitude_damping(out_f, g0, 0.5)["nbt"].item() == 100)

    # ---- S2: shuffle ----
    cfg = {"infoopt": {"score_shuffle": "stage1", "js_mode": "constraint",
                       "epsilon_rel": 1.0, "temperature": 1.0,
                       "kappa_frac": 0.6},
           "dataset": {"name": "CIFAR100"}, "seed": 100}
    gh = np.ones(10) / 10
    ctl = InfoOptController(cfg, gh)
    rng_before = torch.get_rng_state()
    np_before = np.random.get_state()
    py_before = random.getstate()
    w1, info1 = ctl.compute_weights(make_results(), round_idx=7)
    rec = info1["shuffle_record"]
    check("S2a shuffle record present with permutation and scores",
          rec is not None
          and sorted(rec["permutation"]) == list(range(5))
          and sorted(rec["original_scores"]) == sorted(
              rec["solver_scores"]))
    check("S2b shuffle seed follows the protocol derivation",
          rec["seed_int"] == int.from_bytes(
              hashlib.sha256(b"stage1-score-shuffle-v1|CIFAR100|100|7")
              .digest()[:8], "big"))
    ctl2 = InfoOptController(cfg, gh)
    w2, info2 = ctl2.compute_weights(make_results(), round_idx=7)
    check("S2c deterministic across repeats", w1 == w2 and
          info1["shuffle_record"]["permutation"]
          == info2["shuffle_record"]["permutation"])
    check("S2d training RNG not advanced by shuffled compute_weights",
          torch.equal(rng_before, torch.get_rng_state())
          and _np_state_eq(np_before, np.random.get_state())
          and random.getstate() == py_before)

    # S2e: configured loader behavior is num_workers=2 — verify the full
    # epoch order drawn by a worker-enabled loader is identical before and
    # after the shuffled compute_weights call.
    def epoch_order():
        ds = TensorDataset(torch.arange(16))
        dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)
        return torch.cat([b[0] for b in dl])

    torch.set_rng_state(rng_before)
    order_plain = epoch_order()
    torch.set_rng_state(rng_before)
    ctl3 = InfoOptController(cfg, gh)
    ctl3.compute_weights(make_results(), round_idx=7)
    order_after = epoch_order()
    check("S2e worker-enabled (num_workers=2) epoch order identical "
          "after shuffle", torch.equal(order_plain, order_after))

    # ---- S3: solver ----
    scores = np.array([0.3, -0.2, 0.1, 0.0, 0.2])
    P = np.eye(5) + 1e-3
    P = P / P.sum(axis=1, keepdims=True)
    q = np.ones(5) / 5

    # S3a-i: loose epsilon -> closed-form path, solver NOT called
    w_loose, di_loose = solve_entropic_weights(scores, 1.0, P, q, 10.0)
    z = (scores - scores.max())
    w_closed = np.exp(z) / np.exp(z).sum()
    check("S3a loose eps: closed-form path (solver not called)",
          di_loose["solver_called"] is False
          and di_loose["active"] is False
          and di_loose["feasible"] is True
          and di_loose["fallback_used"] is False
          and np.allclose(w_loose, w_closed, atol=1e-12))

    # S3a-ii: forced constrained case — epsilon strictly below js_free but
    # above the uniform mixture's JS, so SLSQP MUST run and succeed.
    js_free, _ = _js_and_grad(w_closed, P, q)
    js_unif, _ = _js_and_grad(np.ones(5) / 5, P, q)
    eps_tight = 0.5 * js_free
    assert js_unif < eps_tight < js_free, "test fixture infeasible"
    w_con, di_con = solve_entropic_weights(scores, 1.0, P, q, eps_tight)
    check("S3b constrained case: SLSQP invoked, feasible, successful",
          di_con["solver_called"] is True
          and di_con["solver_success"] is True
          and di_con["fallback_used"] is False
          and di_con["active"] is True
          and di_con["feasible"] is True
          and di_con["js"] <= eps_tight * (1.0 + 1e-3)
          and abs(float(w_con.sum()) - 1.0) < 1e-8
          and float(w_con.min()) > 0.0)

    # S3c: forced solver failure via monkeypatch -> flagged fallback
    import scipy.optimize
    class FakeRes:
        success = False
        message = "forced failure (test)"
        x = np.full(5, 0.2)
    orig_minimize = scipy.optimize.minimize
    scipy.optimize.minimize = lambda *a, **k: FakeRes()
    try:
        w_f, di_f = solve_entropic_weights(scores, 1.0, P, q, eps_tight)
    finally:
        scipy.optimize.minimize = orig_minimize
    check("S3c forced failure -> fallback flagged with reason",
          di_f["fallback_used"] is True
          and di_f["solver_called"] is True
          and di_f["solver_success"] is False
          and di_f["fallback_reason"] == "slsqp_failed"
          and di_f["solver_message"] == "forced failure (test)")

    # S3d: non-finite signals raise
    bad = make_results()
    bad[2].mi_proxy = float("nan")
    try:
        ctl3 = InfoOptController(cfg, gh)
        ctl3.compute_weights(bad, round_idx=1)
        check("S3d non-finite signal raises", False)
    except ValueError as e:
        check("S3d non-finite signal raises", "non-finite" in str(e))

    # S3e: zero-norm reference -> null + reason
    sd0 = {"w": torch.zeros(2), "nbt": torch.tensor(0)}
    res0 = make_results(C=2)
    for r in res0:
        r.state = sd0
    diag = build_round_diag(1, "fedavg", None, res0, [0.5, 0.5],
                            {"scores": [0.0, 0.0]}, sd0, sd0,
                            average_states, {"w"})
    check("S3e zero reference norm -> null + reason",
          diag["ratio_actual_over_fedavg_full"] is None
          and diag["null_reason_full"] is not None)

    # ---- S4: diagnostics reject non-finite ----
    res4 = make_results(C=2)
    for r in res4:
        r.state = {"w": torch.ones(2), "nbt": torch.tensor(1)}
    res4[0].state = {"w": torch.tensor([float("nan"), 1.0]),
                     "nbt": torch.tensor(0)}
    try:
        build_round_diag(1, "fedavg", None, res4, [0.5, 0.5],
                         {"scores": [0.0, 0.0]}, sd0,
                         {"w": torch.ones(2), "nbt": torch.tensor(1)},
                         average_states, {"w"})
        check("S4a non-finite update raises", False)
    except ValueError as e:
        check("S4a non-finite update raises", "non-finite" in str(e))

    # ---- S5: preflight subprocess from an unrelated CWD (portable) ----
    test_root = (os.environ.get("STAGE1_TEST_DATA_ROOT")
                 or os.environ.get("FOURARM_TEST_CIFAR_ROOT")
                 or None)
    default_root = rstr(Path(__file__).resolve().parents[1] / "datasets")
    if test_root is None and Path(default_root).is_dir():
        test_root = default_root
    bash_exe = _find_bash()
    if bash_exe is None:
        check("S5 preflight subprocess exits 0 from unrelated CWD", False,
              "no bash found (Git Bash on Windows / system bash elsewhere)")
    elif test_root is None or not Path(test_root).is_dir():
        skip("S5 preflight subprocess exits 0 from unrelated CWD",
             "SKIP: set STAGE1_TEST_DATA_ROOT to a CIFAR-100 root")
        skip("S5b schedules only in override dir; no run dirs", "SKIP")
    else:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp = root / "exp"
            shared = root / "shared"
            shared.mkdir()
            env = dict(os.environ, PKG=str(HERE), EXP_ROOT=str(exp),
                       DATASETS="cifar100",
                       DATASETS_ROOT=test_root,
                       SHARED_DIR=str(shared), PREFLIGHT_ONLY="1",
                       PY=sys.executable)
            launcher = HERE / "launch_stage1_gpu1.sh"
            r = subprocess.run([bash_exe, _msys(launcher)],
                               cwd="/" if sys.platform.startswith("win")
                               else tempfile.gettempdir(),
                               env=env, capture_output=True, text=True)
            check("S5 preflight subprocess exits 0 from unrelated CWD",
                  r.returncode == 0 and "PREFLIGHT COMPLETE"
                  in r.stdout + r.stderr,
                  (r.stdout + r.stderr).strip().splitlines()[-1][:80])
            check("S5b schedules only in override dir; no run dirs",
                  (shared / "SHARED_MANIFEST.json").is_file()
                  and not list((exp / "runs").glob("*")))


def S6_S7():
    """S6: init-weight identity. S7: deterministic probe context."""
    from utils.stage1_diagnostics import (init_state_hash,
                                          enforce_init_identity)
    from methods.client import train_client
    import methods.client as clientmod

    def tiny_model(seed_w):
        torch.manual_seed(seed_w)
        import importlib.util, types
        pkg = types.ModuleType("sm"); pkg.__path__ = []
        sys.modules["sm"] = pkg
        for mod in ("aib", "cnn"):
            spec = importlib.util.spec_from_file_location(
                f"sm.{mod}", HERE / "snapshot" / "models" / f"{mod}.py")
            m = importlib.util.module_from_spec(spec)
            sys.modules[f"sm.{mod}"] = m
            spec.loader.exec_module(m)
        return sys.modules["sm.cnn"].SmallCNN(num_classes=10,
                                              input_channels=3,
                                              use_aib=False)

    # S6a: hash covers bytes — identical init -> identical hash
    m1 = tiny_model(7); m2 = tiny_model(7)
    check("S6a identical init -> identical canonical hash",
          init_state_hash(m1) == init_state_hash(m2))
    # S6b: changed weights, same seed -> different hash
    m3 = tiny_model(7)
    with torch.no_grad():
        p0 = next(m3.parameters())
        p0.add_(1e-3)
    h1, h3 = init_state_hash(m1), init_state_hash(m3)
    check("S6b changed initial weights -> different hash (same seed)",
          h1 != h3)
    with tempfile.TemporaryDirectory() as td:
        enforce_init_identity(td, "CIFAR100", 100, h1, "run_a")
        enforce_init_identity(td, "CIFAR100", 100, h1, "run_b")
        try:
            enforce_init_identity(td, "CIFAR100", 100, h3, "run_c")
            check("S6c registry aborts on mismatched init", False)
        except RuntimeError as e:
            check("S6c registry aborts on mismatched init",
                  "init-weight mismatch" in str(e))

    # ---- S7: probe context ----
    base_ds = StochasticImages()
    idx = list(range(64))

    def make_cfg(extra_infoopt=None, num_workers=0):
        io = {"probe_size": 32, "probe_batch_size": 16}
        io.update(extra_infoopt or {})
        return {"dataset": {"name": "CIFAR100", "num_classes": 10},
                "seed": 100,
                "training": {"lr": 0.01, "momentum": 0.0,
                             "weight_decay": 0.0, "batch_size": 16,
                             "local_epochs": 1, "clip_grad_norm": 5.0,
                             "num_workers": num_workers},
                "infoopt": io}

    def run_client(compute_signals, method="fedavg", extra_infoopt=None,
                   num_workers=0):
        m = tiny_model(7)
        torch.manual_seed(4242); np.random.seed(4242); random.seed(4242)
        res = train_client(3, m, base_ds, idx,
                           make_cfg(extra_infoopt, num_workers), "cpu",
                           method=method,
                           compute_info_signals=compute_signals,
                           round_idx=7)
        state_after = (random.getstate(), np.random.get_state(),
                       torch.get_rng_state().clone())
        draws_after = (random.random(), np.random.rand(4),
                       torch.randn(4))
        return res, state_after, draws_after

    def states_equal(sa, sb):
        return all(torch.equal(sa[k], sb[k]) for k in sa)

    # S7a: repeatability — two identical signal-enabled runs
    r1, s1_, d1_ = run_client(True)
    r2, s2_, d2_ = run_client(True)
    check("S7a probe signals reproducible (same derived seed)",
          r1.entropy == r2.entropy and r1.mi_proxy == r2.mi_proxy)

    # S7b: the meaningful restoration test — signal-ENABLED vs
    # signal-DISABLED from identical starting states must produce identical
    # trained client states AND identical subsequent RNG (full python /
    # numpy / torch states and actual draws).
    r_on, st_on, dr_on = run_client(True)
    r_off, st_off, dr_off = run_client(False)
    check("S7b signal collection leaves training invariant "
          "(enabled == disabled: state, RNG, draws)",
          states_equal(r_on.state, r_off.state)
          and r_on.train_loss == r_off.train_loss
          and st_on[0] == st_off[0]
          and _np_state_eq(st_on[1], st_off[1])
          and torch.equal(st_on[2], st_off[2])
          and dr_on[0] == dr_off[0]
          and np.array_equal(dr_on[1], dr_off[1])
          and torch.equal(dr_on[2], dr_off[2]))

    # S7c: arm-independence — verify the PRODUCTION seed derivation against
    # an independent SHA-256 reference, capture the actual seeded RNG state
    # inside the probe context, and confirm round/client change the context
    # while the arm does not.
    ref_seed = int.from_bytes(
        hashlib.sha256(b"stage1-probe-v1|CIFAR100|100|7|3").digest()[:8],
        "big")
    prod_seed = clientmod.stage1_probe_seed("CIFAR100", 100, 7, 3)
    check("S7c-i production probe seed matches independent SHA-256 "
          "reference; round/client change the context",
          prod_seed == ref_seed
          and clientmod.stage1_probe_seed("CIFAR100", 100, 8, 3) != ref_seed
          and clientmod.stage1_probe_seed("CIFAR100", 100, 7, 4) != ref_seed)

    # expected torch RNG state immediately after production seeding
    torch.manual_seed(ref_seed % (2**63 - 1))
    expected_probe_state = torch.get_rng_state().clone()

    arm_cfgs = [("fedavg", {}),
                ("infoopt", {"score_shuffle": ""}),
                ("shuffled_infoopt", {"score_shuffle": "stage1"})]
    captured_states = []
    probe_draws = []
    orig_probe = clientmod.probe_signals

    def spy_probe(model, dataset, indices, device, **kw):
        # runs inside the seeded probe context, before the real probe
        captured_states.append(torch.get_rng_state().clone())
        probe_draws.append(torch.randn(8))
        return orig_probe(model, dataset, indices, device, **kw)

    clientmod.probe_signals = spy_probe
    try:
        arm_results = []
        for method_lbl, extra in arm_cfgs:
            arm_results.append(
                run_client(True, method=method_lbl, extra_infoopt=extra))
    finally:
        clientmod.probe_signals = orig_probe
    check("S7c-ii probe-context RNG state equals freshly-seeded reference "
          "and is identical across arm contexts",
          len(captured_states) == 3
          and all(torch.equal(expected_probe_state, s)
                  for s in captured_states))
    check("S7c-iii probe draws and signals identical across arm contexts",
          len(probe_draws) == 3
          and all(torch.equal(probe_draws[0], pd) for pd in probe_draws[1:])
          and all(r.entropy == arm_results[0][0].entropy
                  and r.mi_proxy == arm_results[0][0].mi_proxy
                  for r, _, _ in arm_results[1:])
          and all(states_equal(arm_results[0][0].state, r.state)
                  for r, _, _ in arm_results[1:]))

    # S7d: genuine worker-enabled training (configured num_workers=2) is
    # reproducible across repeats from identical starting states.
    rw1, _, _ = run_client(True, num_workers=2)
    rw2, _, _ = run_client(True, num_workers=2)
    check("S7d worker-enabled (num_workers=2) training reproducible",
          states_equal(rw1.state, rw2.state)
          and rw1.train_loss == rw2.train_loss)


def S8():
    """S8: client alignment — per-client EMA across rounds and
    histogram/score correspondence under the shuffle."""
    import methods.infoopt as iomod
    from methods.infoopt import InfoOptController

    cfg8 = {"infoopt": {"score_shuffle": "stage1", "js_mode": "constraint",
                        "epsilon_rel": 1.0, "temperature": 1.0,
                        "kappa_frac": 0.0, "signal_ema_beta": 0.5},
            "dataset": {"name": "CIFAR100"}, "seed": 100}
    gh = np.ones(10) / 10

    # S8a: per-client EMA stores across multiple rounds, with the same
    # clients appearing in different positions across rounds.
    ctl = InfoOptController(cfg8, gh)
    beta = 0.5
    rnd1 = [FakeResult(2, 100, (0.40, 0.10, 0.05, 0.01),
                       {"w": torch.ones(2)}),
            FakeResult(5, 100, (0.80, 0.20, 0.05, 0.01),
                       {"w": torch.ones(2)}),
            FakeResult(7, 100, (0.10, 0.30, 0.05, 0.01),
                       {"w": torch.ones(2)}),
            FakeResult(9, 100, (0.20, 0.40, 0.05, 0.01),
                       {"w": torch.ones(2)})]
    ctl.compute_weights(rnd1, round_idx=1)
    rnd2 = [FakeResult(9, 100, (0.60, 0.50, 0.05, 0.01),
                       {"w": torch.ones(2)}),
            FakeResult(5, 100, (0.20, 0.60, 0.05, 0.01),
                       {"w": torch.ones(2)}),
            FakeResult(2, 100, (0.90, 0.70, 0.05, 0.01),
                       {"w": torch.ones(2)}),
            FakeResult(7, 100, (0.30, 0.80, 0.05, 0.01),
                       {"w": torch.ones(2)})]
    ctl.compute_weights(rnd2, round_idx=2)
    exp_mi = {2: beta * 0.40 + (1 - beta) * 0.90,
              5: beta * 0.80 + (1 - beta) * 0.20,
              7: beta * 0.10 + (1 - beta) * 0.30,
              9: beta * 0.20 + (1 - beta) * 0.60}
    exp_ent = {2: beta * 0.10 + (1 - beta) * 0.70,
               5: beta * 0.20 + (1 - beta) * 0.60,
               7: beta * 0.30 + (1 - beta) * 0.80,
               9: beta * 0.40 + (1 - beta) * 0.50}
    check("S8a per-client EMA across rounds matches hand computation",
          set(ctl._mi_ema) == {2, 5, 7, 9}
          and all(abs(ctl._mi_ema[c] - exp_mi[c]) < 1e-12 for c in exp_mi)
          and all(abs(ctl._ent_ema[c] - exp_ent[c]) < 1e-12
                  for c in exp_ent))

    # S8b: under the shuffle, the solver must receive histograms in the
    # ORIGINAL client order while scores arrive permuted. Spy on the
    # production solver entry point.
    recorded = {}
    orig_solve = iomod.solve_entropic_weights

    def spy_solve(scores, tau, P=None, q=None, epsilon=None):
        recorded["scores"] = np.asarray(scores, float).copy()
        recorded["P"] = None if P is None else np.asarray(P, float).copy()
        return orig_solve(scores, tau, P, q, epsilon)

    ctl_b = InfoOptController(cfg8, gh)
    res_b = make_results(C=5)
    iomod.solve_entropic_weights = spy_solve
    try:
        _, info_b = ctl_b.compute_weights(res_b, round_idx=3)
    finally:
        iomod.solve_entropic_weights = orig_solve
    perm = info_b["shuffle_record"]["permutation"]
    exp_P = np.stack([r.label_hist + 1e-12 for r in res_b])
    exp_P = exp_P / exp_P.sum(axis=1, keepdims=True)
    check("S8b histograms unpermuted, scores permuted at solver entry",
          recorded["P"] is not None
          and np.allclose(recorded["P"], exp_P, atol=1e-10)
          and np.allclose(recorded["scores"],
                          np.array(
                              info_b["shuffle_record"]["solver_scores"]),
                          atol=1e-12)
          and np.allclose(
              recorded["scores"],
              np.array(info_b["shuffle_record"]["original_scores"])[perm],
              atol=1e-12))


def S9():
    """S9: failure-injection tests for the verifier gates, the analysis
    verification chain, and the POSIX packager."""
    import methods.infoopt as iomod
    from utils.stage1_diagnostics import (build_round_diag,
                                          init_state_hash)
    from utils.aggregation import (average_states,
                                   apply_magnitude_damping)

    NCLIENTS, CPR, ROUNDS = 10, 5, 100
    SEED = 100
    rng = random.Random(SEED)
    schedule = [rng.sample(range(NCLIENTS), CPR) for _ in range(ROUNDS)]
    client_indices = {str(i): list(range(i * 20, (i + 1) * 20))
                      for i in range(NCLIENTS)}

    def tiny_model(seed_w):
        torch.manual_seed(seed_w)
        import importlib.util, types
        pkg = types.ModuleType("s9m"); pkg.__path__ = []
        sys.modules["s9m"] = pkg
        for mod in ("aib", "cnn"):
            spec = importlib.util.spec_from_file_location(
                f"s9m.{mod}", HERE / "snapshot" / "models" / f"{mod}.py")
            m = importlib.util.module_from_spec(spec)
            sys.modules[f"s9m.{mod}"] = m
            spec.loader.exec_module(m)
        return sys.modules["s9m.cnn"].SmallCNN(num_classes=100,
                                               input_channels=3,
                                               use_aib=False)

    COMMON = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
              "local_epochs": 2, "batch_size": 64, "clip_grad_norm": 5.0,
              "rounds": 100, "clients_per_round": 5}

    def build_valid_run(root, arm="shuffled_infoopt", shuffle_rows=False,
                        acc100=None, run_path=None):
        root = Path(root)
        shared = root / "shared"
        (shared / "init_hashes").mkdir(parents=True)
        run = Path(run_path) if run_path is not None else root / "run"
        run.mkdir(parents=True)
        (shared / f"partition_cifar100_seed{SEED}.json").write_text(
            json.dumps({"dataset": "cifar100", "seed": SEED,
                        "num_clients": NCLIENTS, "alpha": 0.3,
                        "client_indices": client_indices}),
            encoding="utf-8")
        (shared / f"cohort_schedule_cifar100_seed{SEED}.json").write_text(
            json.dumps({"dataset": "cifar100", "seed": SEED,
                        "rounds": ROUNDS, "clients_per_round": CPR,
                        "schedule": schedule}), encoding="utf-8")
        model = tiny_model(7)
        ih = init_state_hash(model)
        (shared / "init_hashes" / f"CIFAR100_seed{SEED}.json").write_text(
            json.dumps({"init_state_sha256": ih, "dataset": "CIFAR100",
                        "seed": SEED}), encoding="utf-8")
        torch.save(model.state_dict(), run / "final_model.pt")
        (run / "init_state_hash.json").write_text(
            json.dumps({"init_state_sha256": ih, "dataset": "CIFAR100",
                        "seed": SEED}), encoding="utf-8")
        rules = {"shuffled_infoopt": ("InfoOpt-Full", "stage1", None),
                 "fedavg": ("fedavg", None, None),
                 "damping_050": ("fedavg", None, 0.5)}[arm]
        method, shuf, gamma = rules
        cfg = {"dataset": {"name": "CIFAR100", "num_classes": 100},
               "seed": SEED, "model": {"use_aib": False},
               "training": dict(COMMON, uniform_agg=False),
               "infoopt": {"score_shuffle": shuf,
                           "early_stop": {"enabled": False}},
               "counterfactual_probe": {"enabled": False}}
        (run / "config_used.json").write_text(json.dumps(
            {"config": cfg, "method": method, "gamma": gamma}),
            encoding="utf-8")
        with open(run / "metrics.csv", "w", encoding="utf-8",
                  newline="") as f:
            import csv as _csv
            wcsv = _csv.writer(f)
            wcsv.writerow(["round", "method", "test_acc", "test_loss",
                           "selected_clients", "early_stop_triggered"])
            rows_data = []
            for r in range(1, ROUNDS + 1):
                acc = acc100 if (acc100 is not None and r == 100) else 0.5
                rows_data.append([r, method, acc, 1.0,
                                  ";".join(map(str, schedule[r - 1])),
                                  "False"])
            if shuffle_rows:
                random.Random(999).shuffle(rows_data)
            wcsv.writerows(rows_data)
        (run / "summary.json").write_text(
            json.dumps({"early_stop_triggered": False}), encoding="utf-8")
        (run / "partition_used.json").write_text(
            json.dumps(client_indices), encoding="utf-8")

        g0 = {"w": torch.zeros(2), "nbt": torch.tensor(0)}
        lines = []
        for rnd in range(1, ROUNDS + 1):
            cids = schedule[rnd - 1]
            results = [FakeResult(c, 20, (0.1, 0.2, 0.05, 0.01),
                                  {"w": torch.tensor([float(c), 1.0]),
                                   "nbt": torch.tensor(c)})
                       for c in cids]
            if arm in ("shuffled_infoopt",):
                weights = [0.3, 0.2, 0.2, 0.15, 0.15]
                scores = [0.3, -0.2, 0.1, 0.0, 0.2]
                seed_int = iomod.stage1_shuffle_seed("CIFAR100", SEED, rnd)
                perm = [int(i) for i in
                        np.random.default_rng(seed_int).permutation(5)]
                info = {"scores": scores, "temperature_used": 1.0,
                        "mixture_js": 0.001, "epsilon": 0.01,
                        "constraint_feasible": True,
                        "constraint_active": False, "nu": 0.0,
                        "solver_calls": 1,
                        "solver_final_status": {"called": False,
                                                "success": None,
                                                "message": None},
                        "fallback_used": False, "fallback_reason": None,
                        "shuffle_record": {
                            "round": rnd, "seed_int": seed_int,
                            "permutation": perm,
                            "original_scores": scores,
                            "solver_scores": [scores[p] for p in perm]}}
            else:
                weights = [0.2] * 5
                info = None
            new_state = average_states([r.state for r in results],
                                       weights)
            if gamma is not None:
                new_state = apply_magnitude_damping(new_state, g0, gamma)
            d = build_round_diag(rnd, method, gamma, results, weights,
                                 info, g0, new_state, average_states,
                                 {"w"})
            lines.append(json.dumps(d, allow_nan=False))
        (run / "diagnostics.jsonl").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
        return run, shared

    def run_verifier(run, shared, arm):
        return subprocess.run(
            [sys.executable, str(HERE / "verify_run.py"),
             "--run-dir", str(run), "--dataset", "cifar100",
             "--seed", str(SEED), "--arm", arm,
             "--schedules-dir", str(shared)],
            capture_output=True, text=True)

    with tempfile.TemporaryDirectory() as td:
        run, shared = build_valid_run(td)
        r = run_verifier(run, shared, "shuffled_infoopt")
        check("S9a synthetic valid run passes the full verifier",
              r.returncode == 0,
              (r.stdout + r.stderr).strip().splitlines()[-1][:100])

    def expect_reject(mutate, name, needle):
        with tempfile.TemporaryDirectory() as td:
            run, shared = build_valid_run(td)
            mutate(run, shared)
            r = run_verifier(run, shared, "shuffled_infoopt")
            check(name, r.returncode != 0
                  and needle in (r.stdout + r.stderr),
                  (r.stdout + r.stderr).strip()[:100])

    def mut_weight(run, shared):
        p = run / "diagnostics.jsonl"
        rows = [json.loads(x) for x in
                p.read_text(encoding="utf-8").splitlines()]
        rows[4]["weights"][0] += 0.05
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n",
                     encoding="utf-8")
    expect_reject(mut_weight,
                  "S9b tampered weights rejected (recomputed gates)", "")

    def mut_perm(run, shared):
        p = run / "diagnostics.jsonl"
        rows = [json.loads(x) for x in
                p.read_text(encoding="utf-8").splitlines()]
        perm = rows[0]["shuffle_permutation"]
        perm[0], perm[1] = perm[1], perm[0]
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n",
                     encoding="utf-8")
    expect_reject(mut_perm,
                  "S9c wrong shuffle permutation rejected",
                  "correspondence")

    def mut_nan(run, shared):
        p = run / "diagnostics.jsonl"
        rows = [json.loads(x) for x in
                p.read_text(encoding="utf-8").splitlines()]
        rows[2]["weight_deviation_l1"] = float("nan")
        # default json.dumps emits a bare NaN token (non-strict JSON)
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n",
                     encoding="utf-8")
    expect_reject(mut_nan,
                  "S9d bare NaN in diagnostics rejected (strict JSON)",
                  "malformed")

    def mut_init(run, shared):
        p = run / "init_state_hash.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["init_state_sha256"] = "0" * 64
        p.write_text(json.dumps(rec), encoding="utf-8")
    expect_reject(mut_init,
                  "S9e init hash != registry rejected", "init")

    def mut_cos(run, shared):
        p = run / "diagnostics.jsonl"
        rows = [json.loads(x) for x in
                p.read_text(encoding="utf-8").splitlines()]
        rows[3]["cosine_actual_fedavg_full"] = None
        rows[3]["null_reason_full"] = None
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n",
                     encoding="utf-8")
    expect_reject(mut_cos,
                  "S9f null cosine without reason rejected",
                  "without reason")

    def mut_solver(run, shared):
        p = run / "diagnostics.jsonl"
        rows = [json.loads(x) for x in
                p.read_text(encoding="utf-8").splitlines()]
        rows[0]["fallback_used"] = True
        rows[0]["fallback_reason"] = None
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n",
                     encoding="utf-8")
    expect_reject(mut_solver,
                  "S9g inconsistent solver fields rejected", "fallback")

    # --- analysis verification chain ---
    def write_shared_manifest(shared):
        entries = []
        for s in (100, 101, 102, 103, 104):
            pp = shared / f"partition_cifar100_seed{s}.json"
            sp = shared / f"cohort_schedule_cifar100_seed{s}.json"
            if not pp.is_file():
                pp.write_text(json.dumps(
                    {"dataset": "cifar100", "seed": s, "num_clients": 10,
                     "alpha": 0.3, "client_indices": client_indices}),
                    encoding="utf-8")
            if not sp.is_file():
                r2 = random.Random(s)
                sp.write_text(json.dumps(
                    {"dataset": "cifar100", "seed": s, "rounds": 100,
                     "clients_per_round": 5,
                     "schedule": [r2.sample(range(10), 5)
                                  for _ in range(100)]}),
                    encoding="utf-8")
            entries.append({
                "partition": {"path": str(pp),
                              "sha256": hashlib.sha256(
                                  pp.read_bytes()).hexdigest()},
                "schedule": {"path": str(sp),
                             "sha256": hashlib.sha256(
                                 sp.read_bytes()).hexdigest()}})
        (shared / "SHARED_MANIFEST.json").write_text(json.dumps(
            {"generated_by": "test", "seeds": [100, 101, 102, 103, 104],
             "files": entries}), encoding="utf-8")

    ARMS6 = ("fedavg", "uniform", "infoopt", "shuffled_infoopt",
             "damping_050", "damping_075")
    DS3 = ("cifar100", "eurosat", "pathmnist")

    def manifest_cells(status="pending", dupe=False):
        cells = [{"dataset": ds, "arm": a, "seed": s, "status": status,
                  "reason": "", "run_dir": None,
                  "verification_report": None, "hashes": {},
                  "terminal_acc": None, "failed_attempts": []}
                 for ds in DS3 for a in ARMS6
                 for s in (100, 101, 102, 103, 104)]
        if dupe:
            cells.append(dict(cells[0]))
        return cells

    def run_analysis(exp_root, shared):
        env = dict(os.environ, EXP_ROOT=str(exp_root),
                   SHARED_DIR=str(shared))
        return subprocess.run(
            [sys.executable, str(HERE / "analyze_stage1.py")],
            capture_output=True, text=True, env=env)

    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"; exp.mkdir()
        shared = Path(td) / "shared"; shared.mkdir()
        write_shared_manifest(shared)
        (exp / "RUN_MANIFEST.json").write_text(json.dumps(
            {"cells": manifest_cells("pending")}), encoding="utf-8")
        r = run_analysis(exp, shared)
        check("S9h analysis aborts on pending cells",
              r.returncode != 0 and "pending" in r.stderr + r.stdout,
              (r.stderr + r.stdout).strip().splitlines()[-1][:100])

    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"; exp.mkdir()
        shared = Path(td) / "shared"; shared.mkdir()
        write_shared_manifest(shared)
        (exp / "RUN_MANIFEST.json").write_text(json.dumps(
            {"cells": manifest_cells("pending", dupe=True)}),
            encoding="utf-8")
        r = run_analysis(exp, shared)
        check("S9i analysis aborts on duplicate cell keys",
              r.returncode != 0 and "duplicate" in r.stderr + r.stdout,
              (r.stderr + r.stdout).strip().splitlines()[-1][:100])

    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"; exp.mkdir()
        shared = Path(td) / "shared"; shared.mkdir()
        write_shared_manifest(shared)
        (exp / "analysis_stage1.json").write_text("{}", encoding="utf-8")
        (exp / "RUN_MANIFEST.json").write_text(json.dumps(
            {"cells": manifest_cells("pending")}), encoding="utf-8")
        r = run_analysis(exp, shared)
        check("S9j analysis refuses to overwrite existing output",
              r.returncode != 0
              and "overwrite" in r.stderr + r.stdout,
              (r.stderr + r.stdout).strip().splitlines()[-1][:100])

    # --- S9l/m/n: analysis gate failure-injection ---
    EVIDENCE7 = ("config_used.json", "metrics.csv", "summary.json",
                 "final_model.pt", "partition_used.json",
                 "diagnostics.jsonl", "init_state_hash.json")

    def run_hashes(run):
        return {fn: hashlib.sha256((run / fn).read_bytes()).hexdigest()
                for fn in EVIDENCE7}

    def full_manifest(exp, run=None, hashes=None, status_override=None,
                      invalid_reason="documented test failure"):
        cells = []
        for ds in DS3:
            for a in ARMS6:
                for s in (100, 101, 102, 103, 104):
                    if run is not None and (ds, a, s) == \
                            ("cifar100", "shuffled_infoopt", 100):
                        cells.append({"dataset": ds, "arm": a, "seed": s,
                                      "status": "complete", "reason": "",
                                      "run_dir": str(run),
                                      "verification_report": None,
                                      "hashes": hashes,
                                      "terminal_acc": 0.5,
                                      "failed_attempts": []})
                    else:
                        rep = Path(exp) / "verification" / \
                            f"{ds}_{a}_seed{s}.json"
                        rep.parent.mkdir(parents=True, exist_ok=True)
                        rep.write_text(json.dumps(
                            {"verified": False,
                             "failures": ["documented test failure"]}),
                            encoding="utf-8")
                        cells.append({
                            "dataset": ds, "arm": a, "seed": s,
                            "status": status_override or "invalid",
                            "reason": invalid_reason, "run_dir": None,
                            "verification_report": str(rep),
                            "hashes": {}, "terminal_acc": None,
                            "failed_attempts": []})
        return {"cells": cells}

    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"; exp.mkdir()
        run, shared = build_valid_run(td)
        write_shared_manifest(shared)
        h = run_hashes(run)
        del h["summary.json"]
        (exp / "RUN_MANIFEST.json").write_text(json.dumps(
            full_manifest(exp, run, h)), encoding="utf-8")
        r = run_analysis(exp, shared)
        check("S9l-i analysis aborts on incomplete evidence-hash set",
              r.returncode != 0
              and "incomplete evidence-hash set" in r.stderr + r.stdout,
              (r.stderr + r.stdout).strip().splitlines()[-1][:100])

    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"; exp.mkdir()
        run, shared = build_valid_run(td)
        write_shared_manifest(shared)
        h = run_hashes(run)
        h["metrics.csv"] = "z" * 64
        (exp / "RUN_MANIFEST.json").write_text(json.dumps(
            full_manifest(exp, run, h)), encoding="utf-8")
        r = run_analysis(exp, shared)
        check("S9l-ii analysis aborts on invalid SHA-256 value",
              r.returncode != 0
              and "invalid SHA-256" in r.stderr + r.stdout,
              (r.stderr + r.stdout).strip().splitlines()[-1][:100])

    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"; exp.mkdir()
        shared = Path(td) / "shared"; shared.mkdir()
        write_shared_manifest(shared)
        (exp / "RUN_MANIFEST.json").write_text(json.dumps(
            full_manifest(exp, status_override="mystery")),
            encoding="utf-8")
        r = run_analysis(exp, shared)
        check("S9m analysis aborts on unknown cell status",
              r.returncode != 0
              and "unknown cell status" in r.stderr + r.stdout,
              (r.stderr + r.stdout).strip().splitlines()[-1][:100])

    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"; exp.mkdir()
        shared = Path(td) / "shared"; shared.mkdir()
        write_shared_manifest(shared)
        (exp / "RUN_MANIFEST.json").write_text(json.dumps(
            full_manifest(exp, invalid_reason="  ")), encoding="utf-8")
        r = run_analysis(exp, shared)
        check("S9n analysis aborts on undocumented invalid cell",
              r.returncode != 0
              and "without a documented reason" in r.stderr + r.stdout,
              (r.stderr + r.stdout).strip().splitlines()[-1][:100])

    # S9o: shuffled metrics row order — round-100 endpoint everywhere
    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp"
        run, shared = build_valid_run(
            td, shuffle_rows=True, acc100=0.77,
            run_path=exp / "runs"
            / "stage1_cifar100_shuffled_infoopt_seed100_t1")
        r = run_verifier(run, shared, "shuffled_infoopt")
        ok_v = r.returncode == 0 and "terminal_acc=0.7700" in r.stdout
        env = dict(os.environ, EXP_ROOT=str(exp), SHARED_DIR=str(shared))
        rb = subprocess.run(
            [sys.executable, str(HERE / "build_run_manifest.py")],
            capture_output=True, text=True, env=env)
        term = None
        mp = exp / "RUN_MANIFEST.json"
        if mp.is_file():
            mj = json.loads(mp.read_text(encoding="utf-8"))
            for c in mj["cells"]:
                if c["status"] == "complete":
                    term = c["terminal_acc"]
        check("S9o shuffled metrics rows: round-100 endpoint correct in "
              "verifier and manifest",
              ok_v and rb.returncode == 0 and term == 0.77,
              f"verifier_rc={r.returncode} builder_rc={rb.returncode} "
              f"terminal={term}")

    # --- POSIX packager ---
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        (src / "snapshot" / "utils").mkdir(parents=True)
        (src / "snapshot" / "utils" / "x.py").write_text("x = 1\n")
        (src / "top.py").write_text("y = 2\n")
        out = Path(td) / "pkg.zip"
        r = subprocess.run(
            [sys.executable, str(HERE / "package_zip.py"),
             str(src), str(out)], capture_output=True, text=True)
        import zipfile
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            dest = Path(td) / "extract"
            z.extractall(dest)
        check("S9k packager emits POSIX members; plain extractall works",
              r.returncode == 0
              and all("\\" not in n for n in names)
              and "snapshot/utils/x.py" in names
              and (dest / "snapshot" / "utils" / "x.py").is_file())


def main_wrapper():
    main()
    S6_S7()
    S8()
    S9()
    failed = [n for n, ok in RESULTS if not ok]
    passed = [n for n, ok in RESULTS if ok]
    print(f"\nPASSED: {len(passed)}  FAILED: {len(failed)}  "
          f"SKIPPED: {len(SKIPPED)}")
    if SKIPPED:
        print("SKIPPED (not executed, not counted as passed):")
        for n, r in SKIPPED:
            print(f"  - {n}: {r}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print(f"ALL {len(passed)} EXECUTED STAGE1 REGRESSION TESTS PASSED")
    if SKIPPED:
        print("NOTE: launch readiness requires a run with "
              "--require-preflight and zero skips.")


if __name__ == "__main__":
    main_wrapper()
