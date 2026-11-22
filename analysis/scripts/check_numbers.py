#!/usr/bin/env python3
"""Consistency checks for the revised manuscript and supplement.

Usage: python3 scripts/check_numbers.py [BUILD_DIR]
BUILD_DIR (optional) holds main_revised.{aux,log,blg} and supplementary.{aux,log};
when given, cross-document references and LaTeX logs are checked too.

Checks
 1. Primary family: recompute paired-t CI and p from (mean, sd_diff, n); Holm with m=6.
 1b. Revision-5 four-arm experiment: checksums + independent per-seed recomputation.
 1c. Preservation: historical data/ files and revision-5 four-arm record counts are
     byte-identical / unchanged by this revision.
 1d. Revision-7 matched RA-InfoOpt experiment: checksums, cell uniqueness (60 unique
     dataset/seed/arm records), independent per-seed recomputation of the primary
     (six-contrast) and secondary (three-contrast probe) families, contrast direction,
     and family sizes.
 1e. Revision-8 saved-checkpoint CIFAR-100 step-size pilot: checksums, frozen verdict
     strings, per-seed-per-control G-values (adaptive headroom + claude comparator)
     against pilot_decision.json, corrected (v2) exploratory held-out MAE/MSE against
     the repaired exploratory_signal_held_out_corrected.json (not the stale
     pilot_decision.json exploratory block), and independent recomputation of the
     buffer-policy cohort-mean comparisons from cohort_summary.csv.
 2. Generated tables match the verified outputs (Table 3, Table 4, A.1, A.2, A.3, Table 5,
    A.4, Table 6, A.5, A.6).
 3. Every number quoted in the prose matches the verified outputs.
 4. No superseded historical value survives (except listed intentional references).
 5. Hard-coded cross-document references resolve to the right labels.
 6. LaTeX logs: no undefined references/citations, no multiply-defined labels, no errors.
Exit status is non-zero if any check fails.
"""
import csv
import json
import math
import os
import re
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_scan import span_of, find_tokens_outside_spans, count_tokens_in_spans  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(ROOT, *p)
BUILD = sys.argv[1] if len(sys.argv) > 1 else None
MANUSCRIPT_DIR = sys.argv[2] if len(sys.argv) > 2 else (os.environ.get("INFOOPT_MANUSCRIPT_DIR") or ROOT)
MD = lambda *p: os.path.join(MANUSCRIPT_DIR, *p)
fails = 0


def ok(cond, msg):
    global fails
    print(("  OK   " if cond else "  FAIL ") + msg)
    fails += not cond


def strip_comments(t):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in t.splitlines())


def load_text(path, label):
    if not os.path.exists(path):
        print(f"  SKIP {label}: not found at {path}")
        return None
    return open(path, encoding="utf-8").read()


main_raw = load_text(MD("main_revised.tex"), "main manuscript")
supp_raw = load_text(MD("supplementary.tex"), "supplement manuscript")
main = strip_comments(main_raw) if main_raw else ""
supp = strip_comments(supp_raw) if supp_raw else ""
MANUSCRIPT_PRESENT = bool(main and supp)
if not MANUSCRIPT_PRESENT:
    print("  NOTE Manuscript source not present; prose/reference/superseded-value checks will be skipped.")
    print("       Provide the directory containing main_revised.tex and supplementary.tex as the second")
    print("       argument, or set INFOOPT_MANUSCRIPT_DIR.")

tabs = {f: strip_comments(open(D("tables", f)).read()) for f in os.listdir(D("tables")) if f.endswith(".tex")}
cs = json.load(open(D("data", "paired_contrasts.json")))["contrasts"]
prim = {(r["dataset"], r["row"]): r for r in csv.DictReader(open(D("data", "primary_summary.csv")))}
sens = json.load(open(D("data", "sensitivity_latest_repeat.json")))["results"]
strata = json.load(open(D("data", "ra_strata_summaries.json")))["strata"]

# Revision 5: four-arm controlled experiment (kept separate from the historical data/ files above).
fa_cs = json.load(open(D("data_fourarm", "fourarm_primary_contrasts.json")))
fa_ints = json.load(open(D("data_fourarm", "fourarm_aib_interactions.json")))
fa_prim = {(r["dataset"], r["arm"]): r for r in csv.DictReader(open(D("data_fourarm", "fourarm_primary_summary.csv")))}
fa_perseed_rows = list(csv.DictReader(open(D("data_fourarm", "fourarm_per_seed_round100.csv"))))

# Revision 7: matched RA-InfoOpt experiment (kept separate from the historical data/ files
# and from the revision-5 four-arm data_fourarm/ files above).
ra_cs = json.load(open(D("data_ra", "ra_primary_contrasts.json")))
ra_probe_cs = json.load(open(D("data_ra", "ra_probe_contrasts.json")))
ra_prim = {(r["dataset"], r["arm"]): r for r in csv.DictReader(open(D("data_ra", "ra_primary_summary.csv")))}
ra_perseed_rows = list(csv.DictReader(open(D("data_ra", "ra_per_seed_round100.csv"))))

# Revision 8: saved-checkpoint CIFAR-100 step-size pilot (kept separate from every
# other data_*/ directory above). One-step evaluations from saved checkpoints.
stepsize_decision_raw = json.load(open(D("data_stepsize", "stepsize_pilot_decision.json")))


def sg(x, nd=3):
    s = f"{x:+.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


print("1. Primary family recomputation")
est = [c for c in cs if c["estimable"]]
ok(len(cs) == 6 and len(est) == 3, "six contrasts, three estimable")
for c in est:
    n = c["n"]; se = c["sd_diff"] / math.sqrt(n); t = c["mean_diff_pp"] / se
    p = 2 * stats.t.sf(abs(t), n - 1); h = stats.t.ppf(.975, n - 1) * se
    ok(abs(p - c["p_raw"]) < 1e-9, f"{c['dataset']}: p_raw {c['p_raw']:.6f} recomputed {p:.6f}")
    ok(abs(c["mean_diff_pp"] - h - c["ci95"][0]) < 1e-9 and abs(c["mean_diff_pp"] + h - c["ci95"][1]) < 1e-9,
       f"{c['dataset']}: CI recomputed")
order = sorted(est, key=lambda c: c["p_raw"]); run = 0
for k, c in enumerate(order):
    run = min(1.0, max(run, (6 - k) * c["p_raw"]))
    ok(abs(run - c["p_holm"]) < 1e-9, f"{c['dataset']}: Holm(m=6) {run:.6f} = reported {c['p_holm']:.6f}")

print("1b. Four-arm experiment: checksums and independent recomputation from per-seed data")
sys.path.insert(0, D("scripts"))
import make_tables as _mt
_mt.verify_fourarm_package()  # raises on any checksum or byte-identity mismatch
ok(True, "four-arm package checksums verified (frozen package + data_fourarm/ byte-identical copies)")
ok(len(fa_prim) == 12, "four-arm primary summary has 12 rows (4 arms x 3 datasets)")
ok(len(fa_cs) == 6, "four-arm primary family has six contrasts")
ok(len(fa_ints) == 3, "four-arm interaction family has three contrasts")

fa_seed = {}
for r in fa_perseed_rows:
    fa_seed.setdefault((r["dataset"], r["arm"]), {})[int(r["seed"])] = float(r["round100_accuracy_fraction"])
SEEDS = [100, 101, 102, 103, 104]


def paired_t(diffs):
    n = len(diffs); m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n); t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    half = stats.t.ppf(.975, n - 1) * se
    return m, sd, (m - half, m + half), p


def holm(raw_ps):
    order = sorted(range(len(raw_ps)), key=lambda i: raw_ps[i])
    out = [None] * len(raw_ps); run = 0
    for k, i in enumerate(order):
        run = min(1.0, max(run, (len(raw_ps) - k) * raw_ps[i]))
        out[i] = run
    return out


for a in ["fedavg_aiboff", "fedavg_aibon", "infoopt_aiboff", "infoopt_aibon"]:
    for d in ["cifar100", "eurosat", "pathmnist"]:
        vals = [fa_seed[(d, a)][s] * 100 for s in SEEDS]
        m = sum(vals) / 5
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / 4)
        r = fa_prim[(d, a)]
        ok(abs(m - float(r["mean_accuracy_percent"])) < 1e-9 and abs(sd - float(r["sd_accuracy_pp"])) < 1e-9,
           f"four-arm primary summary {d}/{a}: mean/sd recomputed from per-seed data match")

raw6 = []
for d in ["cifar100", "eurosat", "pathmnist"]:
    for aib in ["aiboff", "aibon"]:
        diffs = [fa_seed[(d, f"infoopt_{aib}")][s] * 100 - fa_seed[(d, f"fedavg_{aib}")][s] * 100 for s in SEEDS]
        m, sd, ci, p = paired_t(diffs)
        c = [x for x in fa_cs if x["dataset"] == d and x["aib"] == aib][0]
        ok(abs(m - c["mean_difference_pp"]) < 1e-9 and abs(sd - c["sd_difference_pp"]) < 1e-9
           and abs(ci[0] - c["ci95_low_pp"]) < 1e-9 and abs(ci[1] - c["ci95_high_pp"]) < 1e-9
           and abs(p - c["p_raw"]) < 1e-9, f"four-arm contrast {d}/{aib}: mean/sd/CI/p_raw recomputed match")
        raw6.append((d, aib, p))
holm6 = holm([p for _, _, p in raw6])
for (d, aib, p), h in zip(raw6, holm6):
    c = [x for x in fa_cs if x["dataset"] == d and x["aib"] == aib][0]
    ok(abs(h - c["p_holm"]) < 1e-9, f"four-arm contrast {d}/{aib}: Holm(m=6) recomputed {h:.6f} = reported {c['p_holm']:.6f}")

raw3 = []
for d in ["cifar100", "eurosat", "pathmnist"]:
    diffs = [(fa_seed[(d, "infoopt_aibon")][s] - fa_seed[(d, "infoopt_aiboff")][s]
              - fa_seed[(d, "fedavg_aibon")][s] + fa_seed[(d, "fedavg_aiboff")][s]) * 100 for s in SEEDS]
    m, sd, ci, p = paired_t(diffs)
    c = [x for x in fa_ints if x["dataset"] == d][0]
    ok(abs(m - c["mean_difference_pp"]) < 1e-9 and abs(sd - c["sd_difference_pp"]) < 1e-9
       and abs(ci[0] - c["ci95_low_pp"]) < 1e-9 and abs(ci[1] - c["ci95_high_pp"]) < 1e-9
       and abs(p - c["p_raw"]) < 1e-9, f"four-arm interaction {d}: mean/sd/CI/p_raw recomputed match")
    raw3.append((d, p))
holm3 = holm([p for _, p in raw3])
for (d, p), h in zip(raw3, holm3):
    c = [x for x in fa_ints if x["dataset"] == d][0]
    ok(abs(h - c["p_holm"]) < 1e-9, f"four-arm interaction {d}: Holm(m=3) recomputed {h:.6f} = reported {c['p_holm']:.6f}")
sig_ds = [c["dataset"] for c in fa_ints if c["ci95_low_pp"] > 0 or c["ci95_high_pp"] < 0]
ok(sig_ds == ["cifar100"], "only the CIFAR-100 interaction's 95% CI excludes zero")

print("1c. Preservation of earlier evidence (byte-identical to revision 6)")
preserved = {
    D("data", "primary_summary.csv"): "75793caa5a7de62464016162a25f75bf433b5d69a240f97777d1ec9a3fabfad3",
    D("data", "paired_contrasts.json"): "ae8d2e914211d880b816a1e274161499b71c734f53b934e18dc600c513d3c540",
    D("data", "sensitivity_latest_repeat.json"): "deff8ea6308adb0be3cdca451b2c6a258d4ea81a4ca85401ecfe40e260ccc146",
    D("data", "ra_strata_summaries.json"): "dbb9b783ed3308b4ac81e1e1e4564db06c61d306ac78343b3afb5a7384575c26",
}
import hashlib
for path, h in preserved.items():
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    ok(got == h, f"preserved historical file byte-identical to revision 6: {os.path.relpath(path, ROOT)}")
ok(len(prim) == 36, "historical primary_summary.csv still has 36 rows (3 datasets x 12 arms), unchanged by revision 7")
ok(len(fa_prim) == 12 and len(fa_cs) == 6 and len(fa_ints) == 3,
   "revision-5 four-arm record counts unchanged by revision 7 (12/6/3)")

print("1d. Matched RA-InfoOpt experiment: checksums and independent recomputation from per-seed data")
import make_tables as _mt2
_mt2.verify_ra_package()  # raises on any checksum or byte-identity mismatch
ok(True, "RA-matched package checksums verified (frozen package + data_ra/ byte-identical copies)")
ok(len(ra_prim) == 12, "RA-matched primary summary has 12 rows (4 arms x 3 datasets)")
ok(len(ra_cs) == 6, "RA-matched primary family has six contrasts")
ok(len(ra_probe_cs) == 3, "RA-matched secondary (probe) family has three contrasts")

ra_keys = [(r["dataset"], r["seed"], r["arm"]) for r in ra_perseed_rows]
ok(len(ra_keys) == 60 and len(set(ra_keys)) == 60,
   "RA-matched per_seed_round100.csv has 60 unique (dataset, seed, arm) records (3 x 5 x 4)")

ra_seed = {}
for r in ra_perseed_rows:
    ra_seed.setdefault((r["dataset"], r["arm"]), {})[int(r["seed"])] = float(r["test_acc_percent"])
for a in ["fedavg_aibon", "infoopt_aibon", "ra_probeoff", "ra_probeon"]:
    for d in ["cifar100", "eurosat", "pathmnist"]:
        vals = [ra_seed[(d, a)][s] for s in SEEDS]
        m = sum(vals) / 5
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / 4)
        r = ra_prim[(d, a)]
        ok(abs(m - float(r["mean_percent"])) < 1e-9 and abs(sd - float(r["sd_percent"])) < 1e-9,
           f"RA-matched primary summary {d}/{a}: mean/sd recomputed from per-seed data match")

ra_raw6 = []
for d in ["cifar100", "eurosat", "pathmnist"]:
    for base_arm, label in [("fedavg_aibon", "ra_probeoff minus fedavg_aibon"),
                             ("infoopt_aibon", "ra_probeoff minus infoopt_aibon")]:
        diffs = [ra_seed[(d, "ra_probeoff")][s] - ra_seed[(d, base_arm)][s] for s in SEEDS]
        m, sd, ci, p = paired_t(diffs)
        c = [x for x in ra_cs if x["dataset"] == d and x["contrast"] == label][0]
        ok(abs(m - c["mean_diff_pp"]) < 1e-9 and abs(sd - c["sd_diff_pp"]) < 1e-9
           and abs(ci[0] - c["ci95_pp"][0]) < 1e-9 and abs(ci[1] - c["ci95_pp"][1]) < 1e-9
           and abs(p - c["raw_p"]) < 1e-9, f"RA-matched contrast {d}/{label}: mean/sd/CI/p_raw recomputed match")
        ok(m == c["mean_diff_pp"] and (m > 0) == (c["mean_diff_pp"] > 0),
           f"RA-matched contrast {d}/{label}: direction (ra_probeoff minus control) preserved")
        ra_raw6.append((d, label, p))
ra_holm6 = holm([p for _, _, p in ra_raw6])
for (d, label, p), h in zip(ra_raw6, ra_holm6):
    c = [x for x in ra_cs if x["dataset"] == d and x["contrast"] == label][0]
    ok(abs(h - c["holm_p"]) < 1e-9, f"RA-matched contrast {d}/{label}: Holm(m=6) recomputed {h:.6f} = reported {c['holm_p']:.6f}")
sig_primary = [(c["dataset"], c["contrast"]) for c in ra_cs if c["ci95_pp"][0] > 0 or c["ci95_pp"][1] < 0]
ok(set(sig_primary) == {("cifar100", "ra_probeoff minus fedavg_aibon"), ("eurosat", "ra_probeoff minus infoopt_aibon")},
   "exactly the two documented individual CIs exclude zero (CIFAR-100 RA-FedAvg, EuroSAT RA-InfoOpt-Full)")
ok(min(c["holm_p"] for c in ra_cs) > 0.18 and min(c["holm_p"] for c in ra_cs) < 0.19,
   "smallest Holm-adjusted p-value in the RA-matched primary family is approximately 0.189")
ok(all(c["holm_p"] >= 0.05 for c in ra_cs), "no RA-matched primary contrast survives Holm correction at alpha=0.05")

ra_raw3 = []
for d in ["cifar100", "eurosat", "pathmnist"]:
    diffs = [ra_seed[(d, "ra_probeon")][s] - ra_seed[(d, "ra_probeoff")][s] for s in SEEDS]
    m, sd, ci, p = paired_t(diffs)
    c = [x for x in ra_probe_cs if x["dataset"] == d][0]
    ok(abs(m - c["mean_diff_pp"]) < 1e-9 and abs(sd - c["sd_diff_pp"]) < 1e-9
       and abs(ci[0] - c["ci95_pp"][0]) < 1e-9 and abs(ci[1] - c["ci95_pp"][1]) < 1e-9
       and abs(p - c["raw_p"]) < 1e-9, f"RA-matched probe contrast {d}: mean/sd/CI/p_raw recomputed match")
    ra_raw3.append((d, p))
ra_holm3 = holm([p for _, p in ra_raw3])
for (d, p), h in zip(ra_raw3, ra_holm3):
    c = [x for x in ra_probe_cs if x["dataset"] == d][0]
    ok(abs(h - c["holm_p"]) < 1e-9, f"RA-matched probe contrast {d}: Holm(m=3) recomputed {h:.6f} = reported {c['holm_p']:.6f}")
ok(all(c["ci95_pp"][0] <= 0 <= c["ci95_pp"][1] for c in ra_probe_cs),
   "all three RA-matched probe (instrumentation-sensitivity) intervals include zero")
ok(all(c["holm_p"] >= 0.05 for c in ra_probe_cs), "no RA-matched probe contrast survives Holm correction at alpha=0.05")

print("1e. Saved-checkpoint step-size pilot: checksums and independent recomputation")
_mt.verify_stepsize_package()  # raises on any checksum or byte-identity mismatch
ok(True, "step-size pilot package checksums verified (frozen package + data_stepsize/ byte-identical copies)")
ss_decision = _mt.load_stepsize_decision()  # raises if any frozen verdict string has changed
ok(True, "step-size pilot frozen verdict strings match GO_NO_GO.md v3 "
         "(adaptive_headroom=INSUFFICIENT, claude_comparator=NO-GO, "
         "exploratory_variability_signal=EXPLORATORY_PROBE, "
         "overall_recommendation=ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT)")
ss_expected_headroom = {
    (202, "global"): 0.005736845793547451, (202, "phase"): 0.0029000836407696694,
    (203, "global"): 0.022655942563657442, (203, "phase"): 0.0036493037471065012,
}
ss_expected_claude = {
    (202, "global"): -0.01972243652343749, (202, "phase"): -0.02255919867621527,
    (203, "global"): -0.035514947283709464, (203, "phase"): -0.0545215861002604,
}
for key, v in ss_expected_headroom.items():
    ok(abs(ss_decision["headroom"][key] - v) < 1e-9, f"step-size pilot headroom G {key}: {v:+.6f} (pilot_decision.json)")
for key, v in ss_expected_claude.items():
    ok(abs(ss_decision["claude"][key] - v) < 1e-9, f"step-size pilot claude-comparator G {key}: {v:+.6f} (pilot_decision.json)")
ok(ss_decision["headroom_gate_failure_reason"] == "screen_unsatisfied_against_both_controls_on_at_least_one_held_out_seed",
   "step-size pilot headroom gate_failure_reason matches frozen record")
ok(ss_decision["claude_gate_failure_reason"] == "at_least_one_seed_or_control_failed_claude_screen",
   "step-size pilot claude_comparator gate_failure_reason matches frozen record")
# seed 203 fails H2 (phases-positive >= 2 of 3) against both controls -- the specific
# sub-criterion quoted in the supplement's "four findings" paragraph.
ok(stepsize_decision_raw["details"]["adaptive_headroom"]["per_seed_per_control"]["203"]["global"]["n_positive_phases"] == 1
   and stepsize_decision_raw["details"]["adaptive_headroom"]["per_seed_per_control"]["203"]["phase"]["n_positive_phases"] == 1,
   "step-size pilot: seed 203 has only 1 of 3 phases positive against both fixed controls (quoted in supplement)")

ss_exploratory = _mt.load_stepsize_exploratory()  # raises if fit_status != 'fit' (i.e. uses the corrected, not stale, file)
ss_expected_exploratory = {
    "phase_only": (0.009036155845108362, 0.00031357235306608236),
    "phase_plus_variability": (0.020016339820771462, 0.0006961795674210221),
}
for pid, (mae, mse) in ss_expected_exploratory.items():
    ok(abs(ss_exploratory[pid]["mae_held"] - mae) < 1e-12 and abs(ss_exploratory[pid]["mse_held"] - mse) < 1e-12,
       f"step-size pilot corrected exploratory {pid}: held MAE {mae:.9f} / MSE {mse:.9f} "
       f"(exploratory_signal_held_out_corrected.json, not the stale pilot_decision.json block)")
ok(ss_exploratory["phase_only"]["n_dev_cells_used"] == 27 and ss_exploratory["phase_only"]["n_held_cells_used"] == 54
   and ss_exploratory["phase_plus_variability"]["n_dev_cells_used"] == 27
   and ss_exploratory["phase_plus_variability"]["n_held_cells_used"] == 54,
   "step-size pilot corrected exploratory: 27 dev cells / 54 held-out cells for both predictors")
# The stale pilot_decision.json exploratory_variability_signal block must remain fit_status=no_cells
# (the documented pre-correction defect); the manuscript must not be sourced from it.
ok(stepsize_decision_raw["details"]["exploratory_variability_signal"]["verdict"] == "EXPLORATORY_PROBE",
   "step-size pilot: pilot_decision.json exploratory verdict is EXPLORATORY_PROBE (not a deployability claim)")

ss_buffer = _mt.load_stepsize_buffer()  # independently recomputed from cohort_summary.csv raw rows
ss_expected_buffer = {
    "A_freeze_buffers_at_gamma1": (-0.06291166246202258, -0.07046892541956018),
    "B_valsel_vs_full_gamma1": (0.0646421095106337, 0.068213816550926),
    "B_valsel_vs_params_only_gamma1": (0.12755377197265627, 0.1386827419704862),
    "C_valsel_vs_fixed_global": (0.014196394178602448, 0.009915021882233813),
    "C_valsel_vs_fixed_phase": (0.0032746936939380853, 0.00037071397569444077),
}
for comp, (held, dev) in ss_expected_buffer.items():
    ok(abs(ss_buffer[comp]["held_mean"] - held) < 1e-9 and abs(ss_buffer[comp]["dev_mean"] - dev) < 1e-9,
       f"step-size pilot buffer-policy {comp}: held {held:+.6f} (n=18) / dev {dev:+.6f} (n=9), "
       f"independently recomputed from cohort_summary.csv")

print("2. Generated tables vs verified outputs")
t3 = tabs["table3_retrospective.tex"]
for (ds, row), r in prim.items():
    if int(r["n"]) > 0:
        cell = f"{float(r['mean']):.2f} $\\pm$ {float(r['sd']):.2f} ({r['n']})"
        ok(cell in t3, f"Table 3 {ds}/{row}: {cell}")
ok(t3.count("\\emph{withheld}") == 3, "Table 3: three RA-InfoOpt cells withheld")
ok(t3.count("\\emph{excluded}") == 1 and int(prim[("cifar100", "fedopt")]["n"]) == 0, "Table 3: CIFAR-100 FedOpt excluded (n=0 in primary)")
sys.path.insert(0, D("scripts"))
import make_tables as _mt
_fo = _mt.load_excluded_cell_sensitivity()  # verifies package checksums and recomputes mean/sd
ok(len(_fo) == 1 and (_fo[0].dataset, _fo[0].row) == ("cifar100", "fedopt"), "FedOpt sensitivity loaded from checksum-verified frozen package")
fo = {"mean": _fo[0].mean, "sd": _fo[0].sd, "n": _fo[0].n}
fo_txt = f"{fo['mean']:.2f} $\\pm$ {fo['sd']:.2f} ({fo['n']})"
fedopt_row = [l for l in t3.splitlines() if l.startswith("FedOpt &")][0]
ok(fo_txt in t3 and "44.72" not in fedopt_row and "\\emph{excluded}" in fedopt_row,
   f"Table 3: CIFAR-100 FedOpt sensitivity {fo_txt} only in footnote; primary cell excluded")
if MANUSCRIPT_PRESENT:
    for name, txt in [("main_revised.tex", main), ("supplementary.tex", supp)]:
        ok(re.search(r"44\.72\s*\$\\pm\$\s*2\.15", txt) is not None, f"{name}: FedOpt sensitivity quoted as 44.72 +/- 2.15")
else:
    print("  SKIP FedOpt prose-quote check (manuscript not present)")
for f in ["table4_primary_contrasts.tex", "tableA_primary.tex"]:
    for c in est:
        toks = [sg(c["mean_diff_pp"]), sg(c["ci95"][0]), sg(c["ci95"][1]), f"{c['p_raw']:.3f}", f"{c['p_holm']:.3f}",
                sg(c["mean_diff_pp"] / c["sd_diff"], 2)]
        ok(all(x in tabs[f] for x in toks), f"{f} {c['dataset']}: {', '.join(toks)}")
    ok(tabs[f].count("\\emph{not estimable") == 3, f"{f}: three RA contrasts not estimable")
ta2 = tabs["tableA_sensitivity.tex"]
for r in sens:
    if "-" in r["row"]:
        toks = [sg(r["sensitivity_mean_latest_diff"]), sg(r["ci95"][0]), sg(r["ci95"][1]), f"{r['p_raw']:.3f}"]
    else:
        toks = [f"{r['primary_mean_earliest']:.3f}", f"{r['sensitivity_mean_latest']:.3f}", sg(r["delta"])]
    ok(all(x in ta2 for x in toks), f"Table A.2 {r['dataset']}/{r['row']}: {', '.join(toks)}")
ta3 = tabs["tableA_ra_strata.tex"]
for s in strata:
    v = f"{s['mean']:.2f} $\\pm$ {s['sd']:.2f}" if s["sd"] is not None else f"{s['mean']:.2f} (single run)"
    ok(s["stratum"][:6] in ta3 and v in ta3, f"Table A.3 stratum {s['stratum'][:6]}: {v}")

t5 = tabs["table5_fourarm.tex"]
for (d, a), r in fa_prim.items():
    cell = f"{r['mean_accuracy_percent']} $\\pm$ {r['sd_accuracy_pp']}"
    # table rounds to 2dp; recompute the exact rounded string instead
    cell = f"{float(r['mean_accuracy_percent']):.2f} $\\pm$ {float(r['sd_accuracy_pp']):.2f} ({r['n']})"
    ok(cell in t5, f"Table 5(a) {d}/{a}: {cell}")
for c in fa_cs:
    toks = [sg(c["mean_difference_pp"]), sg(c["ci95_low_pp"]), sg(c["ci95_high_pp"]),
            f"{c['p_raw']:.3f}", f"{c['p_holm']:.3f}"]
    ok(all(x in t5 for x in toks), f"Table 5(b) {c['dataset']}/{c['aib']}: {', '.join(toks)}")
ta4 = tabs["tableA4_fourarm_interaction.tex"]
for c in fa_ints:
    toks = [sg(c["mean_difference_pp"]), sg(c["ci95_low_pp"]), sg(c["ci95_high_pp"]),
            f"{c['p_raw']:.4f}", f"{c['p_holm']:.4f}"]
    ok(all(x in ta4 for x in toks), f"Table A.4 {c['dataset']}: {', '.join(toks)}")

t6 = tabs["table6_ra_matched.tex"]
for (d, a), r in ra_prim.items():
    cell = f"{float(r['mean_percent']):.2f} $\\pm$ {float(r['sd_percent']):.2f} ({r['n']})"
    ok(cell in t6, f"Table 6(a) {d}/{a}: {cell}")
for c in ra_cs:
    toks = [sg(c["mean_diff_pp"]), sg(c["ci95_pp"][0]), sg(c["ci95_pp"][1]), f"{c['raw_p']:.4f}", f"{c['holm_p']:.4f}"]
    ok(all(x in t6 for x in toks), f"Table 6(b) {c['dataset']}/{c['contrast']}: {', '.join(toks)}")
ta5 = tabs["tableA5_ra_probe.tex"]
for c in ra_probe_cs:
    toks = [sg(c["mean_diff_pp"]), sg(c["ci95_pp"][0]), sg(c["ci95_pp"][1]), f"{c['raw_p']:.4f}", f"{c['holm_p']:.4f}"]
    ok(all(x in ta5 for x in toks), f"Table A.5 {c['dataset']}: {', '.join(toks)}")

ta6 = tabs["tableA6_stepsize_pilot.tex"]
for key, v in ss_expected_headroom.items():
    ok(sg(v, 6) in ta6, f"Table A.6 headroom G {key}: {sg(v, 6)}")
for key, v in ss_expected_claude.items():
    ok(sg(v, 6) in ta6, f"Table A.6 claude-comparator G {key}: {sg(v, 6)}")

print("3. Numbers quoted in prose")
if MANUSCRIPT_PRESENT:
    quoted_main = ["+0.822", "-4.733", "+0.092", "[-0.612,+2.256]", "[-9.714,+0.248]", "[-1.379,+1.563]",
                   "p=0.187", "p=0.934", "p=0.058", "p=0.346", "p=0.871", "p=1.000", "-4.770", "[-10.231,+0.690]",
                   "p=0.072", "-0.072", "[-2.977,+2.832]", "p=0.948", "-0.037", "-0.164", "0.164", "165", "135",
                   # revision 5: four-arm experiment, six primary contrasts + three interactions
                   "+0.536", "[-0.359,+1.431]", "-0.184", "[-1.249,+0.881]", "-4.156", "[-10.230,+1.919]",
                   "-3.741", "[-8.490,+1.008]", "-1.565", "[-5.225,+2.094]", "-0.304", "[-0.923,+0.316]",
                   "-0.720", "[-1.108,-0.332]", "p=0.0067", "p=0.0202", "+0.415", "[-1.984,+2.814]",
                   "+1.262", "[-2.885,+5.408]"]
    flat = re.sub(r"[\s$~]", "", main)
    for q in quoted_main:
        ok(q.replace(" ", "") in flat, f"main text quotes {q}")
    truth = {"+0.822": est[0]["mean_diff_pp"], "-4.733": [c for c in est if c["dataset"] == "eurosat"][0]["mean_diff_pp"],
             "+0.092": [c for c in est if c["dataset"] == "pathmnist"][0]["mean_diff_pp"]}
    for k, v in truth.items():
        ok(f"{v:+.3f}" == k, f"quoted {k} equals verified {v:.6f}")
    sc = {r["dataset"]: r for r in sens if "-" in r["row"]}
    ok(f"{sc['eurosat']['sensitivity_mean_latest_diff']:+.3f}" == "-4.770" and f"{sc['pathmnist']['sensitivity_mean_latest_diff']:+.3f}" == "-0.072",
       "quoted sensitivity contrasts equal verified values")
    ok(abs(max(abs(r["delta"]) for r in sens if "-" not in r["row"]) - 0.1643452) < 1e-6, "max |movement| = 0.164 pp (verified)")

    for c in fa_cs:
        q = f"{c['mean_difference_pp']:+.3f}"
        ok(q in flat.replace(" ", ""), f"main text quotes four-arm contrast {c['dataset']}/{c['aib']}: {q}")
    for c in fa_ints:
        q = f"{c['mean_difference_pp']:+.3f}"
        ok(q in flat.replace(" ", ""), f"main text quotes four-arm interaction {c['dataset']}: {q}")

    ra_quoted_main = ["+0.208", "[+0.030,+0.386]", "+6.681", "[+0.428,+12.935]", "0.189", "0.206"]
    for q in ra_quoted_main:
        ok(q.replace(" ", "") in flat, f"main text quotes RA-matched value {q}")
    ra_cifar_fedavg = [c for c in ra_cs if c["dataset"] == "cifar100" and c["contrast"] == "ra_probeoff minus fedavg_aibon"][0]
    ra_euro_infoopt = [c for c in ra_cs if c["dataset"] == "eurosat" and c["contrast"] == "ra_probeoff minus infoopt_aibon"][0]
    ok(f"{ra_cifar_fedavg['mean_diff_pp']:+.3f}" == "+0.208" and f"{ra_euro_infoopt['mean_diff_pp']:+.3f}" == "+6.681",
       "main text's two highlighted RA-matched values equal verified contrasts")
    ok(abs(min(c["holm_p"] for c in ra_cs) - 0.1888193276109172) < 1e-6,
       "main text's 'approximately 0.189' equals the verified smallest Holm-adjusted p-value in the RA-matched primary family")
    supp_flat = re.sub(r"[\s$~]", "", supp)
    for q in ["-4.770", "-0.072", "-4.733", "+0.092", "-0.164", "42.65,80.81,and87.03", "43.14,80.03,and85.61", "40.05", "43.20"]:
        ok(q in supp_flat, f"supplement text quotes {q}")
    for q in ["537", "four_arm_20261009"]:
        ok(q.replace("_", "\\_") in supp or q in supp, f"supplement text quotes RA-matched protocol detail {q}")
    ok(re.search(r"counterfactual probe", supp) is not None and re.search(r"probe\s+model", supp) is not None,
       "supplement describes the counterfactual probe and its instrumentation-sensitivity caveat")
    ok(re.search(r"ten\s+supplied\s+files|eleven\s+supplied\s+files", supp) is not None,
       "supplement documents the RA-matched analysis-package file count")
    ok(f"{float(prim[('cifar100','gamma0.75')]['mean']):.2f}" == "42.65" and f"{float(prim[('pathmnist','fedavg')]['mean']):.2f}" == "85.61",
       "supplement damping values equal verified summaries")

    ok(re.search(r"headroom.{0,400}INSUFFICIENT", supp, re.S) is not None,
       "supplement states the step-size pilot's adaptive-headroom verdict (INSUFFICIENT)")
    ok(re.search(r"NO-GO", supp) is not None, "supplement states the step-size pilot's claude-comparator verdict (NO-GO)")
    for q in ["0.009036", "0.0003136", "0.020016", "0.0006962"]:
        ok(q in supp_flat, f"supplement quotes corrected exploratory value {q}")
    ok("0.0629" in supp_flat and "0.0705" in supp_flat, "supplement quotes buffer-freeze held-out/dev loss-improvement values")
    ok("77 of 90" in supp or "77of90" in supp_flat, "supplement quotes the export-defect affected-source count (77 of 90)")
    ok("154 of 900" in supp or "154of900" in supp_flat, "supplement quotes the export-defect checked-cell mismatch count (154 of 900)")
    ok("9 of" in supp and "81" in supp, "supplement quotes the partial-recovery source counts (9 recovered / 81 unrecoverable)")
    ok(re.search(r"partition seed 42", supp) is not None, "supplement states the pilot's fixed partition seed (42)")
    ok(re.search(r"seed[s]? \(?201\)?", supp) is not None and "202" in supp and "203" in supp,
       "supplement states the pilot's development (201) and held-out (202, 203) training seeds")
    ok(re.search(r"rounds 5, 50, and 100|5, 50 and 100", supp) is not None,
       "supplement states the pilot's three saved-checkpoint rounds (5, 50, 100)")
    ok(re.search(r"later change record", supp) is not None and re.search(r"PRE_EXECUTION_AMENDMENT", supp.replace("\\_", "_")) is not None,
       "supplement distinguishes PRE_EXECUTION_AMENDMENT.md as a later change record, not certified original pre-execution bytes")
    for name, txt in [("main_revised.tex", main), ("supplementary.tex", supp)]:
        ok(re.search(r"step-size pilot|step-size control", txt) is not None, f"{name}: mentions the step-size pilot")
    ok(re.search(r"analyzed separately and does not change either", main) is not None or re.search(r"kept separate", main) is not None,
       "main text keeps the step-size pilot separate from the four-arm and matched RA-InfoOpt experiments")
    ok(re.search(r"analyzed\s+separately\s+and\s+does\s+not\s+change", supp) is not None,
       "supplement states the step-size pilot is analyzed separately and does not change either experiment's estimates")
    ok(re.search(r"does not bear on", main) is None and re.search(r"does not bear on", supp) is None,
       "neither document still uses the ambiguous 'does not bear on' wording for the step-size pilot")
    ok("onwhichthehistoricalcomparisonresultsarebased" in flat,
       "main Section 4 opening scopes the retrospective reanalysis to the historical comparison results only")
    ok("frozenbeforelaunch,ratherthanselectingamongalready-recordedruns" in flat,
       "main Section 4 opening distinguishes the prospective four-arm and matched RA experiments")
else:
    print("  SKIP all prose-quote checks (manuscript not present)")

print("4. Superseded-value scan")
if MANUSCRIPT_PRESENT:
    orig_main_raw = load_text(MD("original", "main_elsevier_ORIGINAL.tex"), "original main manuscript")
    orig_supp_raw = load_text(MD("original", "Supplementary_Material_ORIGINAL.tex"), "original supplement manuscript")
    ORIGINAL_PRESENT = bool(orig_main_raw and orig_supp_raw)
    if not ORIGINAL_PRESENT:
        print("  SKIP superseded-value scan (original manuscript files not present)")
    else:
        orig_main = orig_main_raw
        orig_supp = orig_supp_raw
        old = set(re.findall(r"\b\d{2}\.\d{2}\b", orig_main[orig_main.index("label{tab:canonical}"):orig_main.index("end{tabular}", orig_main.index("label{tab:canonical}"))]))
        old |= {"0.52", "4.63", "1.43", "0.178", "0.678", "0.531", "0.339", "0.089", "0.265", "10.38", "1.84", "4.50", "0.048", "0.238",
                "43.80", "77.41", "86.82", "37.31", "53.74", "49.15", "4.22", "0.096"}
        verified = {f"{float(r['mean']):.2f}" for r in prim.values() if r["mean"]}
        superseded = sorted(old - verified)
        intentional = {("supplementary.tex", "0.048"),
                       # 44.72 is the author-supplied CIFAR-100 FedOpt SENSITIVITY value (footnote only)
                       ("table3_retrospective.tex", "44.72"), ("main_revised.tex", "44.72"), ("supplementary.tex", "44.72")}
        contrast_tokens = {"0.52", "4.63", "1.43", "0.178", "0.678", "0.531", "0.339", "0.089", "0.265", "10.38", "1.84", "4.50",
                           "0.048", "0.238", "4.22", "0.096"}

        # Revision 6: FedDyn/SCAFFOLD repair-campaign descriptive means (campaign_summary.json,
        # independently re-verified against the audit package's SHA256SUMS.txt). These six values
        # numerically reproduce the original withdrawn FedDyn/SCAFFOLD table row to 2 decimals. A
        # plain file-wide exception would let them (or a copy-pasted figure) reappear anywhere in
        # the document, including in primary-result prose, without the scanner objecting. Instead
        # each token is allowed only inside its designated descriptive-audit passage; an occurrence
        # anywhere else in the file still fails the check (see scripts/span_scan.py).
        campaign_tokens = {"29.10", "42.28", "74.84", "85.10", "85.58", "87.92"}
        main_campaign_span = span_of(main,
            r"\paragraph{Repair-campaign run identification (new in revision 6).}",
            r"\paragraph{Repeat-selection sensitivity.}")
        supp_campaign_span = span_of(supp,
            r"\paragraph{FedDyn/SCAFFOLD repair-campaign audit (new in revision 6).}",
            r"\end{itemize}")
        # 87.92 also has two pre-existing, unrelated legitimate occurrences in the supplement: the
        # "earlier version ... withdrawn claims" sentence (pre-dates revision 6) and the Record of
        # Revisions audit table row. Both are given their own spans rather than a blanket exception.
        supp_withdrawn_claim_span = span_of(supp,
            "The earlier version of this subsection stated that SCAFFOLD and FedDyn",
            "CIFAR-100 FedOpt is excluded from the primary summary under the strict configured-early-stop rule")
        supp_revision_record_span = span_of(supp,
            "A.2: SCAFFOLD/FedDyn converge stably",
            "CIFAR-100 FedOpt excluded.")
        campaign_spans = {
            "main_revised.tex": [main_campaign_span],
            "supplementary.tex": [supp_campaign_span, supp_withdrawn_claim_span, supp_revision_record_span],
        }

        for name, txt in [("main_revised.tex", main), ("supplementary.tex", supp)] + list(tabs.items()):
            for tok in superseded:
                if name == "table3_retrospective.tex" and tok in contrast_tokens:
                    continue  # Table 3 holds means/SDs only; old contrast tokens there would be coincidental SD digits
                if tok in campaign_tokens:
                    spans = campaign_spans.get(name, [])
                    bad = find_tokens_outside_spans(txt, tok, spans)
                    n_ok = count_tokens_in_spans(txt, tok, spans)
                    if bad:
                        ok(False, f"superseded value {tok} in {name} appears outside its designated "
                                  f"descriptive-audit passage ({len(bad)} stray occurrence(s), "
                                  f"{n_ok} correctly inside it)")
                    elif n_ok:
                        print(f"  note {name}: '{tok}' appears {n_ok} time(s), only inside its "
                              f"designated revision-6 repair-campaign descriptive-audit passage "
                              f"(historical descriptive evidence, not a primary result; coincides "
                              f"numerically with the withdrawn table row)")
                    continue
                if re.search(r"(?<![\d.])" + re.escape(tok) + r"(?!\d)", txt):
                    if (name, tok) in intentional:
                        why = "author-supplied sensitivity-only value (footnote/text), not a primary result" if tok == "44.72" else "intentional reference to a withdrawn claim"
                        print(f"  note {name}: '{tok}' appears only as an {why}")
                    else:
                        ok(False, f"superseded value {tok} in {name}")
        ok(True, f"scanned {len(superseded)} superseded values")
else:
    print("  SKIP superseded-value scan (manuscript not present)")

if BUILD:
    if not MANUSCRIPT_PRESENT:
        print("  SKIP cross-document reference and LaTeX-log checks (manuscript not present)")
    else:
        print("5. Cross-document references")

        def labels(aux):
            out = {}
            for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", open(aux).read()):
                out[m.group(1)] = m.group(2)
            return out
        ml = labels(os.path.join(BUILD, "main_revised.aux"))
        sl = labels(os.path.join(BUILD, "supplementary.aux"))
        want_s = {"app:tab:sensitivity": "A.2", "app:tab:ra_strata": "A.3", "fig:repeat_sensitivity": "A.1", "fig:ra_strata": "A.2",
                  "app:sensitivity": "A.4", "app:diagnostics": "D", "app:record": "G", "app:tab:extended_comparison": "E.1",
                  "app:tab:primary_holm": "A.1",
                  "app:fourarm": "A.7", "app:tab:fourarm_interaction": "A.4",
                  "app:ra_matched": "A.8", "app:tab:ra_probe": "A.5",
                  "app:stepsize_pilot": "A.9", "app:tab:stepsize_pilot": "A.6"}
        for k, v in want_s.items():
            ok(sl.get(k) == v, f"supplement {k} = {v} (aux: {sl.get(k)})")
        want_m = {"subsec:provenance": "4.5", "tab:canonical": "3", "tab:paired": "4", "fig:framework": "1",
                  "fig:evaluation_framework": "2", "fig:accuracy": "3", "fig:paired": "4", "subsec:grasp": "3.4",
                  "sec:grasp_provenance": "4.3",
                  "sec:fourarm": "5.4", "tab:fourarm": "5", "fig:fourarm": "5",
                  "sec:ra_matched": "5.5", "tab:ra_matched": "6",
                  "subsec:discussion_diagnostics": "6.2"}
        for k, v in want_m.items():
            ok(ml.get(k) == v, f"main {k} = {v} (aux: {ml.get(k)})")
        print("6. LaTeX logs")
        for stem in ["main_revised", "supplementary"]:
            log = open(os.path.join(BUILD, stem + ".log"), errors="replace").read()
            ok("undefined" not in log.lower().replace("undefined references", "") or not re.search(r"(Reference|Citation) .* undefined", log),
               f"{stem}: no undefined references or citations")
            ok("multiply defined" not in log, f"{stem}: no multiply-defined labels")
            ok(not re.search(r"^! ", log, re.M), f"{stem}: no LaTeX errors")
            print(f"  info {stem}: {len(re.findall('Overfull', log))} overfull box(es)")
        blg = os.path.join(BUILD, "main_revised.blg")
        if os.path.exists(blg):
            ok("Warning--I didn't find" not in open(blg).read(), "BibTeX: all cited keys found")

print("\nRESULT:", "ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
