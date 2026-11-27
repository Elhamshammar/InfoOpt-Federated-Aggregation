#!/usr/bin/env python3
"""MLJ-R4 statistical verification and new retrospective analyses.

Inputs (read-only):
  reproduction/outputs/MLJ_A1_1/seed_level_effects.csv   (12 matched InfoOpt settings, 5 seeds each)
  FedDisco per-seed final accuracies (Supplementary Table F.1; see note below)

Outputs (written to --out):
  MLJ_R4_PAIRED_VERIFICATION.csv     paired contrasts, CI, d_z, raw p, recomputed TOST for S01-S12
  MLJ_R4_EXACT_MDE.csv               exact noncentral-t 80%-power detectable differences
  MLJ_R4_FEDDISCO_TOST.csv           FedDisco prespecified test plus retrospective TOST
  MLJ_R4_TEST_RESULTS.txt            verification test log

Note on FedDisco precision. CIFAR-100 has 10,000 test images, so test accuracy in percent
is an integer multiple of 0.01. The two-decimal values in Supplementary Table F.1 are therefore
exact final accuracies, provided the full test set is used. The script checks this by
reproducing the archived SD (0.2934), CI (-0.3183, 0.4103) and p (0.743585).
"""
import argparse, csv, math, sys
import numpy as np
from scipy import stats
from scipy.optimize import brentq

N, DF, ALPHA, POWER, MARGIN = 5, 4, 0.05, 0.80, 0.5

def paired(d):
    d = np.asarray(d, float); n = len(d)
    m, s = d.mean(), d.std(ddof=1); se = s / math.sqrt(n)
    t95 = stats.t.ppf(0.975, n - 1); t90 = stats.t.ppf(0.95, n - 1)
    t = m / se; p = 2 * stats.t.sf(abs(t), n - 1)
    p_low = stats.t.sf((m + MARGIN) / se, n - 1)    # H0: mean <= -margin
    p_high = stats.t.cdf((m - MARGIN) / se, n - 1)  # H0: mean >= +margin
    return dict(n=n, mean=m, sd=s, se=se, ci95_lo=m - t95 * se, ci95_hi=m + t95 * se,
                t=t, p_two_sided=p, dz=m / s, ci90_lo=m - t90 * se, ci90_hi=m + t90 * se,
                p_low=p_low, p_high=p_high, p_tost=max(p_low, p_high),
                equivalent=bool(m - t90 * se > -MARGIN and m + t90 * se < MARGIN))

def power_two_sided(ncp, df=DF, alpha=ALPHA):
    tc = stats.t.ppf(1 - alpha / 2, df)
    lower = stats.nct.cdf(-tc, df, ncp)
    return stats.nct.sf(tc, df, ncp) + (0.0 if np.isnan(lower) else lower)

def exact_ncp(df=DF, alpha=ALPHA, power=POWER):
    return brentq(lambda d: power_two_sided(d, df, alpha) - power, 0.5, 8, xtol=1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-csv", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.seed_csv)))
    settings = {}
    for r in rows:
        settings.setdefault(r["setting_id"], {"meta": r, "d": []})["d"].append(float(r["diff_pp"]))
    ncp = exact_ncp(); tc = stats.t.ppf(0.975, DF); tq = stats.t.ppf(POWER, DF)
    out_p, out_m = [], []
    for sid in sorted(settings):
        S = settings[sid]; meta = S["meta"]; res = paired(S["d"])
        lab = dict(setting_id=sid, study={"Stage1": 3, "Stage3": 4, "Stage4": 5}.get(meta["stage"], meta["stage"]),
                   dataset=meta["dataset"], model=meta["model"], alpha=meta["alpha"])
        out_p.append({**lab, **{k: res[k] for k in ["n", "mean", "sd", "ci95_lo", "ci95_hi", "t", "p_two_sided", "dz",
                                                   "ci90_lo", "ci90_hi", "p_low", "p_high", "p_tost", "equivalent"]}})
        mde = ncp * res["sd"] / math.sqrt(N)
        out_m.append({**lab, "n": N, "df": DF, "alpha_two_sided": ALPHA, "target_power": POWER,
                      "observed_sd_pp": res["sd"], "noncentrality_parameter": ncp, "mde80_exact_pp": mde,
                      "mde80_approx_pp": (tc + tq) * res["sd"] / math.sqrt(N),
                      "power_at_mde_check": power_two_sided(mde * math.sqrt(N) / res["sd"]),
                      "observed_abs_mean_pp": abs(res["mean"])})
    fd = {"seed": [100, 101, 102, 103, 104],
          "fedavg": [41.84, 43.70, 42.49, 44.88, 42.75], "uniform": [41.91, 43.50, 42.41, 44.69, 43.13],
          "feddisco": [41.95, 43.39, 42.39, 44.99, 42.88], "shuffled": [41.83, 43.49, 42.69, 44.50, 42.86]}
    dd = np.round(np.array(fd["feddisco"]) - np.array(fd["shuffled"]), 10)
    fres = paired(dd); fmde = ncp * fres["sd"] / math.sqrt(N)
    def w(path, recs):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(recs[0])); wr.writeheader(); wr.writerows(recs)
    w(f"{a.out}/MLJ_R4_PAIRED_VERIFICATION.csv", out_p)
    w(f"{a.out}/MLJ_R4_EXACT_MDE.csv", out_m)
    w(f"{a.out}/MLJ_R4_FEDDISCO_TOST.csv", [{"contrast": "FedDisco minus shuffled-discrepancy FedDisco",
        "per_seed_diff_pp": " ".join(f"{x:+.2f}" for x in dd), **{k: fres[k] for k in fres},
        "margin_pp": MARGIN, "mde80_exact_pp": fmde,
        "status": "retrospective, exploratory, nominal, unadjusted; margin carried over from InfoOpt analysis"}])
    # ---------------- verification tests ----------------
    log, fails = [], 0
    def check(name, cond, detail=""):
        nonlocal fails
        log.append(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}"); fails += (not cond)
    rep = {  # values reported in R2.1 manuscript/supplement (Table G.2, TOST table)
        "S01": (0.154, -0.372, 0.680, 0.071), "S02": (-2.422, -8.443, 3.599, 0.787), "S03": (-1.730, -5.753, 2.294, 0.778),
        "S04": (0.196, -0.384, 0.776, 0.110), "S05": (-0.098, -0.592, 0.396, 0.043), "S06": (-2.289, -10.503, 5.925, 0.711),
        "S07": (-2.385, -4.451, -0.320, 0.968), "S08": (-0.237, -4.677, 4.203, 0.439), "S09": (-0.429, -2.084, 1.226, 0.455),
        "S10": (-0.210, -1.377, 0.957, 0.264), "S11": (0.044, -1.533, 1.622, 0.234), "S12": (-0.482, -3.347, 2.383, 0.493)}
    for r in out_p:
        m, lo, hi, pt = rep[r["setting_id"]]
        check(f"{r['setting_id']} mean/CI/TOST reproduce reported values",
              abs(r["mean"] - m) < 6e-4 and abs(r["ci95_lo"] - lo) < 6e-4 and abs(r["ci95_hi"] - hi) < 6e-4 and abs(r["p_tost"] - pt) < 6e-4,
              f"mean={r['mean']:.4f} CI=[{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}] TOST={r['p_tost']:.4f}")
        check(f"{r['setting_id']} TOST p equals max of one-sided p", r["p_tost"] == max(r["p_low"], r["p_high"]))
    eq = [r["setting_id"] for r in out_p if r["equivalent"]]
    check("Only S05 equivalent at +/-0.5 pp", eq == ["S05"], str(eq))
    s05 = [r for r in out_p if r["setting_id"] == "S05"][0]
    check("S05 binding test is the lower-bound test", s05["p_low"] > s05["p_high"], f"p_low={s05['p_low']:.4f} p_high={s05['p_high']:.4f}")
    for r in out_m:
        check(f"{r['setting_id']} power at exact MDE equals 0.80", abs(r["power_at_mde_check"] - 0.8) < 1e-9)
    rng = np.random.default_rng(20261126); sims = 200000
    x = rng.standard_normal((sims, N)) + ncp / math.sqrt(N)
    tt = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(N)); mc = np.mean(np.abs(tt) > tc)
    check("Monte Carlo power at exact ncp approx 0.80", abs(mc - 0.8) < 0.005, f"mc={mc:.4f} ncp={ncp:.6f}")
    check("FedDisco mean reproduces +0.046", abs(fres["mean"] - 0.046) < 1e-9, f"{fres['mean']:.6f}")
    check("FedDisco SD reproduces 0.2934", abs(fres["sd"] - 0.2934) < 5e-5, f"{fres['sd']:.6f}")
    check("FedDisco 95% CI reproduces [-0.3183, 0.4103]", abs(fres["ci95_lo"] + 0.3183) < 5e-5 and abs(fres["ci95_hi"] - 0.4103) < 5e-5,
          f"[{fres['ci95_lo']:.6f},{fres['ci95_hi']:.6f}]")
    check("FedDisco two-sided p reproduces 0.743585", abs(fres["p_two_sided"] - 0.743585) < 5e-7, f"{fres['p_two_sided']:.6f}")
    check("FedDisco TOST matches independent values", abs(fres["ci90_lo"] + 0.233719) < 1e-6 and abs(fres["ci90_hi"] - 0.325719) < 1e-6
          and abs(fres["p_low"] - 0.0070653) < 1e-6 and abs(fres["p_high"] - 0.0129076) < 1e-6,
          f"CI90=[{fres['ci90_lo']:.6f},{fres['ci90_hi']:.6f}] p_low={fres['p_low']:.7f} p_high={fres['p_high']:.7f} p_tost={fres['p_tost']:.7f}")
    check("FedDisco equivalent at +/-0.5 pp", fres["equivalent"])
    check("FedDisco accuracies lie on 0.01 grid (10,000-image test set)",
          all(abs(v * 100 - round(v * 100)) < 1e-9 for k in ["fedavg", "uniform", "feddisco", "shuffled"] for v in fd[k]))
    for k, target in [("fedavg", 43.13), ("uniform", 43.13), ("feddisco", 43.12), ("shuffled", 43.07)]:
        check(f"FedDisco arm mean {k} rounds to {target}", round(float(np.mean(fd[k])), 2) == target, f"{np.mean(fd[k]):.4f}")
    orc = list(csv.DictReader(open(a.oracle_csv)))
    al = np.array([float(r["oracle_nojs_aligned"]) for r in orc]); sh = np.array([float(r["oracle_nojs_shuffled"]) for r in orc])
    fa = np.array([float(r["fedavg"]) for r in orc])
    l_al = np.array([float(r["last10_oracle_aligned"]) for r in orc]); l_sh = np.array([float(r["last10_oracle_shuffled"]) for r in orc])
    o1, o2, o3, o4 = paired(al - sh), paired(al - fa), paired(sh - fa), paired(l_al - l_sh)
    check("Oracle aligned-shuffled +5.300 [1.825, 8.775] p=0.013319 dz=1.894",
          abs(o1["mean"] - 5.300) < 5e-4 and abs(o1["ci95_lo"] - 1.825) < 5e-4 and abs(o1["ci95_hi"] - 8.775) < 5e-4
          and abs(o1["p_two_sided"] - 0.013319) < 5e-6 and abs(o1["dz"] - 1.894) < 5e-4,
          f"{o1['mean']:.4f} [{o1['ci95_lo']:.4f},{o1['ci95_hi']:.4f}] p={o1['p_two_sided']:.6f} dz={o1['dz']:.4f}")
    check("Oracle aligned-FedAvg +1.988 [0.365, 3.611] p=0.027234",
          abs(o2["mean"] - 1.988) < 5e-4 and abs(o2["ci95_lo"] - 0.365) < 5e-4 and abs(o2["p_two_sided"] - 0.027234) < 5e-6,
          f"{o2['mean']:.4f} p={o2['p_two_sided']:.6f}")
    check("Oracle shuffled-FedAvg -3.312 [-8.240, 1.616] p=0.135472",
          abs(o3["mean"] + 3.312) < 5e-4 and abs(o3["ci95_lo"] + 8.240) < 5e-4 and abs(o3["p_two_sided"] - 0.135472) < 5e-6,
          f"{o3['mean']:.4f} p={o3['p_two_sided']:.6f}")
    check("Oracle last-ten-round +5.943 [4.644, 7.243] p=0.000222",
          abs(o4["mean"] - 5.943) < 5e-4 and abs(o4["ci95_lo"] - 4.644) < 5e-4 and abs(o4["p_two_sided"] - 0.000222) < 5e-6,
          f"{o4['mean']:.4f} p={o4['p_two_sided']:.6f}")
    eq_mean = float(np.mean([r["mean"] for r in out_p]))
    check("Equal-setting mean aligned-shuffled -0.824", abs(eq_mean + 0.824) < 5e-4, f"{eq_mean:.4f}")
    check("Nine of 12 setting means negative", sum(r["mean"] < 0 for r in out_p) == 9)
    open(f"{a.out}/MLJ_R4_TEST_RESULTS.txt", "w").write("\n".join(log) + f"\n\n{len(log)-fails}/{len(log)} tests passed\n")
    print("\n".join(log)); print(f"{len(log)-fails}/{len(log)} passed"); sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
