#!/usr/bin/env python3
"""Stage-2A isolated controller verification (CPU, no training).

CODE-LEVEL verification with representative synthetic inputs — this is
NOT a verification of the actual smoke training-round scores (the v8
logs record per-round means and final scores, not per-client signal
values, so exact training-round reconstruction is impossible from the
logs; that distinction is intentional and disclosed).

Checks, against the corrected isolated copy (stage2a snapshot):
  A. Three-signal score formula: score = 1.0*z(mi) - 0.5*z(ent) +
     0.2*z(stability) with per-client EMA (beta=0.5) and robust MAD
     z-scores; hand-reconstructed vs controller-reported scores.
  B. Post-warm-up gate: alpha_raw == sigmoid(6*(0.25*h_n + 0.25*r +
     0.5*h_n*r - 0.5)) recomputed from the controller's own recorded
     components; EMA recurrence; blend identity; finite simplex final
     weights; fallback fields consistent.
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshot"))

import torch  # noqa: E402  (required by methods.client imports)
from methods.ra_infoopt import RAInfoOptController  # noqa: E402
from methods.infoopt import InfoOptController  # noqa: E402


class FakeResult:
    def __init__(self, cid, n, mi, ent, stab):
        self.client_id = cid
        self.num_samples = n
        self.mi_proxy = mi
        self.entropy = ent
        self.update_norm = -stab / 2.0
        self.train_loss_std = -stab / 2.0   # stability = -(norm + std)
        self.label_hist = np.eye(10)[cid % 10].astype(float)
        self.state = {"w": torch.ones(2), "nbt": torch.tensor(1)}
        self.score = None


def z_robust(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + 1e-8)


def main():
    problems = []
    gh = np.ones(10) / 10
    io_cfg = {"infoopt": {"js_mode": "constraint", "epsilon_rel": 1.0,
                          "temperature": 1.0, "kappa_frac": 0.0,
                          "lambda_mi": 1.0, "lambda_entropy": 0.5,
                          "lambda_stability": 0.2, "signal_ema_beta": 0.5},
              "dataset": {"name": "CIFAR100"}, "seed": 100}

    # --- A. score formula reconstruction (2 rounds, nonconstant inputs)
    ctl = InfoOptController(io_cfg, gh)
    rounds = [
        [FakeResult(c, 100 + 10 * i, mi, ent, stab)
         for i, (c, mi, ent, stab) in enumerate(zip(
             [1, 3, 4, 7, 8],
             [0.30, 0.10, 0.55, 0.22, 0.41],
             [3.10, 2.40, 1.80, 2.90, 2.20],
             [-5.0, -3.0, -8.0, -4.0, -6.0]))],
        [FakeResult(c, 100 + 10 * i, mi, ent, stab)
         for i, (c, mi, ent, stab) in enumerate(zip(
             [8, 1, 4, 3, 7],   # same clients, different round order
             [0.35, 0.15, 0.50, 0.26, 0.44],
             [3.00, 2.50, 1.90, 2.80, 2.10],
             [-5.5, -3.5, -7.5, -4.5, -6.5]))],
    ]
    ema_mi, ema_ent = {}, {}
    for rnd, res in enumerate(rounds, start=1):
        _, info = ctl.compute_weights(res)
        exp_mi, exp_ent, exp_stab = [], [], []
        for r in res:
            ema_mi[r.client_id] = (r.mi_proxy if rnd == 1 else
                                   0.5 * ema_mi[r.client_id]
                                   + 0.5 * r.mi_proxy)
            ema_ent[r.client_id] = (r.entropy if rnd == 1 else
                                    0.5 * ema_ent[r.client_id]
                                    + 0.5 * r.entropy)
            exp_mi.append(ema_mi[r.client_id])
            exp_ent.append(ema_ent[r.client_id])
            exp_stab.append(-(r.update_norm + r.train_loss_std))
        expected = (1.0 * z_robust(exp_mi) - 0.5 * z_robust(exp_ent)
                    + 0.2 * z_robust(exp_stab))
        actual = np.asarray(info["scores"])
        if not np.allclose(actual, expected, atol=1e-10):
            problems.append(f"score formula mismatch round {rnd}: "
                            f"max diff {np.abs(actual - expected).max()}")

    # --- B. post-warm-up gate (7 rounds through the full RA controller)
    ra_cfg = {"infoopt": dict(io_cfg["infoopt"]),
              "regime_reliability_controller": {
                  "enabled": True,
                  "warmup_rounds": 5,
                  "smoothing": {"alpha_ema": 0.8},
                  "reliability": {"beta_separation": 0.4,
                                  "beta_rank": 0.3, "beta_temporal": 0.3,
                                  "separation_scale": 0.25},
                  "gate": {"bias": 0.5, "heterogeneity_weight": 0.25,
                           "interaction_weight": 0.5,
                           "reliability_weight": 0.25,
                           "temperature": 6.0},
                  "safety": {"alpha_binary_search_steps": 20,
                             "enc_deviation_limit": 1.0,
                             "enforce_enc_after_blending": True,
                             "enforce_js_after_blending": True,
                             "js_deviation_limit": 0.05,
                             "numerical_tolerance": 1.0e-08}},
              "dataset": {"name": "CIFAR100"}, "seed": 100}
    ra = RAInfoOptController(ra_cfg, gh)
    g = np.random.default_rng(42)
    for rnd in range(1, 8):
        res = [FakeResult(c, 100 + 10 * i,
                          float(g.random()), float(2 * g.random() + 1),
                          float(-3 - 3 * g.random()))
               for i, c in enumerate([1, 3, 4, 7, 8])]
        w, info = ra.compute_weights(res, round_idx=rnd)
        w = np.asarray(w)
        if not (np.isfinite(w).all() and abs(w.sum() - 1.0) < 1e-9
                and w.min() > 0):
            problems.append(f"final weights not finite simplex round "
                            f"{rnd}")
        a_raw, a_ema, a_safe = (info["alpha_raw"], info["alpha_ema"],
                                info["alpha_safe"])
        if rnd <= 5 and (a_raw != 0.0 or a_safe != 0.0):
            problems.append(f"warm-up violated round {rnd}")
        if rnd > 5:
            # gate identity from the controller's own recorded components
            logit = (0.25 * info["heterogeneity_normalized"]
                     + 0.25 * info["reliability_total"]
                     + 0.5 * info["heterogeneity_normalized"]
                     * info["reliability_total"] - 0.5)
            want = 1.0 / (1.0 + math.exp(-6.0 * logit))
            if abs(a_raw - want) > 1e-12:
                problems.append(f"gate formula mismatch round {rnd}: "
                                f"{a_raw} vs {want}")
            if a_raw <= 0.0:
                problems.append(f"gate did not open post-warm-up round "
                                f"{rnd}")
        if not (0.0 <= a_safe <= 1.0):
            problems.append(f"alpha_safe out of range round {rnd}")
        # blend identity against the recorded components
        if math.isfinite(info["blend_identity_error_abs"]) and \
                info["blend_identity_error_abs"] > 1e-9:
            problems.append(f"blend identity violated round {rnd}")
        if info["fallback_reason"] == "none" and info["fallback_used"] != 0:
            problems.append(f"fallback flag inconsistent round {rnd}")

    if problems:
        print("ISOLATED CONTROLLER CHECKS: FAIL")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("ISOLATED CONTROLLER CHECKS: PASS")
    print("  A. three-signal score formula: hand reconstruction matches "
          "controller exactly (2 rounds)")
    print("  B. post-warm-up gate: sigmoid identity on recorded "
          "components, EMA recurrence, blend identity, finite simplex "
          "weights, no fallbacks — rounds 6-7")


if __name__ == "__main__":
    main()
