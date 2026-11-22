#!/usr/bin/env python3
"""Regression tests for the primary/sensitivity separation in make_tables.py.

Run: python3 scripts/test_make_tables.py
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402
import span_scan  # noqa: E402

failures = 0


def check(cond, msg):
    global failures
    print(("PASS " if cond else "FAIL ") + msg)
    failures += not cond


primary = mt.load_primary()
_, sens_summ, _ = mt.load_sensitivity()

# 1. A sensitivity-only value for an excluded primary cell must not change the primary status.
rec = primary[("cifar100", "fedopt")]
check(mt.primary_status(rec) == "excluded", "CIFAR-100 FedOpt primary status is 'excluded'")
tmp = tempfile.mkdtemp()
orig_out = mt.OUT
mt.OUT = tmp
fake = mt.ExcludedCellSensitivity("cifar100", "fedopt", 5, 99.99, 1.0, "test")
mt.table3(primary, [fake])
t3 = open(os.path.join(tmp, "table3_retrospective.tex")).read()
row = [l for l in t3.splitlines() if l.startswith("FedOpt &")][0]
check("\\emph{excluded}" in row and "99.99" not in row, "sensitivity value does not overwrite the primary cell")
check("99.99" in t3, "sensitivity value, when present, is reported separately in the footnote")
mt.table3(primary, [])
t3b = open(os.path.join(tmp, "table3_retrospective.tex")).read()
check("contain no value for this cell" in t3b, "absent sensitivity value is reported as absent, not invented")
real = mt.load_excluded_cell_sensitivity()
mt.table3(primary, real)
t3c = open(os.path.join(tmp, "table3_retrospective.tex")).read()
row = [l for l in t3c.splitlines() if l.startswith("FedOpt &")][0]
check(len(real) == 1 and "\\emph{excluded}" in row and "44.72 $\\pm$ 2.15 (5)" in t3c and "44.72" not in row,
      "supplied FedOpt sensitivity appears only in the footnote; primary cell stays excluded")
try:
    mt.table3(primary, [fake, fake])
    check(False, "duplicate sensitivity-only records raise")
except ValueError:
    check(True, "duplicate sensitivity-only records raise")
mt.OUT = orig_out
shutil.rmtree(tmp)

# 1b. The importer reads the ORIGINAL frozen output and refuses a tampered copy.
check(real and real[0].provenance.endswith("(checksum-verified)") and abs(real[0].mean - 44.724) < 1e-9
      and abs(real[0].sd - 2.153109379478897) < 1e-12 and real[0].n == 5,
      "FedOpt sensitivity read from the frozen package (44.724 +/- 2.1531, n=5)")
tpkg = tempfile.mkdtemp()
shutil.copytree(mt.SENS_PKG, os.path.join(tpkg, "pkg"))
fp = os.path.join(tpkg, "pkg", "outputs", "sensitivity_cifar_fedopt.json")
os.chmod(fp, 0o644)
open(fp, "w").write(open(fp).read().replace("47.33", "47.34"))
try:
    mt.load_excluded_cell_sensitivity(os.path.join(tpkg, "pkg"))
    check(False, "tampered sensitivity file is rejected (checksum)")
except ValueError:
    check(True, "tampered sensitivity file is rejected (checksum)")
for root, dirs, files in os.walk(tpkg):
    for n in dirs + files:
        os.chmod(os.path.join(root, n), 0o755)
shutil.rmtree(tpkg)

# 2. A rule/data conflict must raise rather than resolve silently.
bad = mt.PrimaryRecord("cifar100", "fedopt", 5, 44.0, 1.0, "", "")
try:
    mt.primary_status(bad)
    check(False, "excluded cell with a primary value raises")
except ValueError:
    check(True, "excluded cell with a primary value raises")

# 3. Primary keys are unique.
check(len(primary) == 36, "36 unique primary records (3 datasets x 12 arms)")

# 4. Four-arm package (revision 5): checksums, row/contrast/interaction counts.
mt.verify_fourarm_package()
check(True, "four-arm package checksums verified (frozen package + data_fourarm/ copies)")
fa_primary = mt.load_fourarm_primary()
check(len(fa_primary) == 12, "four-arm primary summary has 12 rows (4 arms x 3 datasets)")
fa_contrasts = mt.load_fourarm_contrasts()
check(len(fa_contrasts) == 6, "four-arm primary family has six within-AIB InfoOpt-FedAvg contrasts")
fa_interactions = mt.load_fourarm_interactions()
check(len(fa_interactions) == 3, "four-arm interaction family has three aggregation x AIB interactions")

# 4b. A tampered working copy must be rejected (byte-identity check against the frozen package).
tpkg2 = tempfile.mkdtemp()
shutil.copytree(mt.FOURARM_WORK, os.path.join(tpkg2, "data_fourarm"))
victim = os.path.join(tpkg2, "data_fourarm", "fourarm_primary_summary.csv")
open(victim, "w").write(open(victim).read().replace("43.102", "43.999"))
orig_work = mt.FOURARM_WORK
mt.FOURARM_WORK = os.path.join(tpkg2, "data_fourarm")
try:
    mt.verify_fourarm_package()
    check(False, "tampered four-arm working copy is rejected (byte-identity check)")
except ValueError:
    check(True, "tampered four-arm working copy is rejected (byte-identity check)")
mt.FOURARM_WORK = orig_work
shutil.rmtree(tpkg2)

# 4c. CIFAR-100 is the only interaction whose 95% CI excludes zero.
sig = [c["dataset"] for c in fa_interactions if c["ci95_low_pp"] > 0 or c["ci95_high_pp"] < 0]
check(sig == ["cifar100"], "only the CIFAR-100 interaction excludes zero at 95% CI")

# 4d. Matched RA-InfoOpt package (revision 7): checksums, row/contrast counts, cell uniqueness.
mt.verify_ra_package()
check(True, "RA-matched package checksums verified (frozen package + data_ra/ copies)")
ra_primary = mt.load_ra_primary()
check(len(ra_primary) == 12, "RA-matched primary summary has 12 rows (4 arms x 3 datasets)")
ra_contrasts = mt.load_ra_contrasts()
check(len(ra_contrasts) == 6, "RA-matched primary family has six RA(probe off)-FedAvg/InfoOpt contrasts")
ra_probe_contrasts = mt.load_ra_probe_contrasts()
check(len(ra_probe_contrasts) == 3, "RA-matched secondary family has three probe-on/off contrasts")

import csv as _csv
ra_rows = list(_csv.DictReader(open(os.path.join(mt.RA_WORK, "ra_per_seed_round100.csv"))))
ra_keys = [(r["dataset"], r["seed"], r["arm"]) for r in ra_rows]
check(len(ra_keys) == 60 and len(set(ra_keys)) == 60,
      "RA-matched per-seed data has 60 unique (dataset, seed, arm) cells (3 x 5 x 4)")

# 4e. A tampered RA-matched working copy must be rejected (byte-identity check).
tpkg3 = tempfile.mkdtemp()
shutil.copytree(mt.RA_WORK, os.path.join(tpkg3, "data_ra"))
victim3 = os.path.join(tpkg3, "data_ra", "ra_primary_summary.csv")
open(victim3, "w").write(open(victim3).read().replace("43.698", "43.999"))
orig_ra_work = mt.RA_WORK
mt.RA_WORK = os.path.join(tpkg3, "data_ra")
try:
    mt.verify_ra_package()
    check(False, "tampered RA-matched working copy is rejected (byte-identity check)")
except ValueError:
    check(True, "tampered RA-matched working copy is rejected (byte-identity check)")
mt.RA_WORK = orig_ra_work
shutil.rmtree(tpkg3)

# 4f. Exactly the two documented individual CIs exclude zero in the RA-matched primary family.
ra_sig = {(c["dataset"], c["contrast"]) for c in ra_contrasts
          if c["ci95_pp"][0] > 0 or c["ci95_pp"][1] < 0}
check(ra_sig == {("cifar100", "ra_probeoff minus fedavg_aibon"), ("eurosat", "ra_probeoff minus infoopt_aibon")},
      "exactly CIFAR-100 RA-FedAvg and EuroSAT RA-InfoOpt-Full exclude zero (individual CIs)")
check(all(c["holm_p"] >= 0.05 for c in ra_contrasts),
      "no RA-matched primary contrast survives Holm correction at alpha=0.05")
check(all(c["ci95_pp"][0] <= 0 <= c["ci95_pp"][1] for c in ra_probe_contrasts),
      "all three RA-matched probe (instrumentation-sensitivity) intervals include zero")

# 4g. Saved-checkpoint step-size pilot (revision 8): checksums and frozen-verdict guard.
mt.verify_stepsize_package()
check(True, "step-size pilot package checksums verified (frozen package + data_stepsize/ copies)")
ss_decision = mt.load_stepsize_decision()
check(set(ss_decision["headroom"].keys()) == {(202, "global"), (202, "phase"), (203, "global"), (203, "phase")},
      "step-size pilot headroom G loaded for both held-out seeds x both fixed controls")
check(set(ss_decision["claude"].keys()) == {(202, "global"), (202, "phase"), (203, "global"), (203, "phase")},
      "step-size pilot claude-comparator G loaded for both held-out seeds x both fixed controls")
check(all(v < 0 for v in ss_decision["claude"].values()),
      "step-size pilot: the tested shrinkage rule is worse than both fixed controls on every evaluated cell")

# 4h. A frozen-verdict change (e.g. a future protocol amendment) must raise, not be silently
# absorbed into the table -- the frozen GO_NO_GO.md v3 vocabulary is a hard check, not a default.
tpkg4 = tempfile.mkdtemp()
shutil.copytree(mt.STEPSIZE_WORK, os.path.join(tpkg4, "data_stepsize"))
victim4 = os.path.join(tpkg4, "data_stepsize", "stepsize_pilot_decision.json")
open(victim4, "w").write(open(victim4).read().replace('"INSUFFICIENT"', '"HEADROOM"'))
orig_ss_work = mt.STEPSIZE_WORK
mt.STEPSIZE_WORK = os.path.join(tpkg4, "data_stepsize")
try:
    mt.load_stepsize_decision()
    check(False, "a changed adaptive_headroom verdict is rejected (frozen-vocabulary guard)")
except ValueError:
    check(True, "a changed adaptive_headroom verdict is rejected (frozen-vocabulary guard)")
mt.STEPSIZE_WORK = orig_ss_work
shutil.rmtree(tpkg4)

# 4i. A tampered step-size working copy must be rejected (byte-identity check).
tpkg5 = tempfile.mkdtemp()
shutil.copytree(mt.STEPSIZE_WORK, os.path.join(tpkg5, "data_stepsize"))
victim5 = os.path.join(tpkg5, "data_stepsize", "stepsize_pilot_decision.json")
open(victim5, "w").write(open(victim5).read().replace("0.005736845793547451", "0.999999999999999"))
mt.STEPSIZE_WORK = os.path.join(tpkg5, "data_stepsize")
try:
    mt.verify_stepsize_package()
    check(False, "tampered step-size working copy is rejected (byte-identity check)")
except ValueError:
    check(True, "tampered step-size working copy is rejected (byte-identity check)")
mt.STEPSIZE_WORK = orig_ss_work
shutil.rmtree(tpkg5)

# 4j. The corrected (v2) exploratory loader must refuse a stale pilot_decision.json-style
# no_cells block, so the manuscript cannot silently be sourced from the pre-correction defect.
ss_exploratory = mt.load_stepsize_exploratory()
check(set(ss_exploratory.keys()) == {"phase_only", "phase_plus_variability"},
      "corrected exploratory loader returns both predictors")
tpkg6 = tempfile.mkdtemp()
shutil.copytree(mt.STEPSIZE_WORK, os.path.join(tpkg6, "data_stepsize"))
victim6 = os.path.join(tpkg6, "data_stepsize", "stepsize_exploratory_held_out_corrected.json")
open(victim6, "w").write(open(victim6).read().replace('"fit_status": "fit"', '"fit_status": "no_cells"', 1))
mt.STEPSIZE_WORK = os.path.join(tpkg6, "data_stepsize")
try:
    mt.load_stepsize_exploratory()
    check(False, "a stale/defective fit_status is rejected by the corrected-exploratory loader")
except ValueError:
    check(True, "a stale/defective fit_status is rejected by the corrected-exploratory loader")
mt.STEPSIZE_WORK = orig_ss_work
shutil.rmtree(tpkg6)

# 4k. Buffer-policy cohort means are independently recomputed from the raw per-cohort CSV
# (not transcribed from report.md), with the trajectory cohort excluded and exact cell counts.
ss_buffer = mt.load_stepsize_buffer()
check(set(ss_buffer.keys()) == {"A_freeze_buffers_at_gamma1", "B_valsel_vs_full_gamma1",
                                 "B_valsel_vs_params_only_gamma1", "C_valsel_vs_fixed_global",
                                 "C_valsel_vs_fixed_phase"},
      "buffer-policy loader returns all five comparisons")
check(ss_buffer["A_freeze_buffers_at_gamma1"]["held_n"] == 18 and ss_buffer["A_freeze_buffers_at_gamma1"]["dev_n"] == 9,
      "buffer-policy cohort-mean cell counts: 18 held-out / 9 dev (trajectory cohort excluded)")
check(ss_buffer["A_freeze_buffers_at_gamma1"]["held_mean"] < 0,
      "buffer-policy: freezing checkpoint buffers at unit scaling worsens held-out loss in this setting")

# 5. Revision 6: the superseded-value scanner's campaign-token exception must be
# position-aware, not a blanket per-file pass. A token whitelisted because it appears
# inside its designated repair-campaign descriptive-audit passage must still be caught
# if it (or a copy-pasted figure) turns up anywhere else in the document, including in
# primary-result prose.
_start, _end = "START_OF_AUDIT_PASSAGE", "END_OF_AUDIT_PASSAGE"
_doc_clean = (
    f"Primary results table shows FedAvg at 85.12 percent.\n"
    f"{_start}\nThe repair-campaign descriptive mean is 29.10 percent (historical evidence only).\n{_end}\n"
    f"Later discussion does not repeat that number.\n"
)
_doc_leaked = (
    f"Primary results table now also reports FedAvg at 29.10 percent (leaked into primary prose).\n"
    f"{_start}\nThe repair-campaign descriptive mean is 29.10 percent (historical evidence only).\n{_end}\n"
)
_span = span_scan.span_of(_doc_clean, _start, _end)
check(span_scan.find_tokens_outside_spans(_doc_clean, "29.10", [_span]) == [],
      "campaign token confined to its descriptive-audit passage is accepted")
_span_leaked = span_scan.span_of(_doc_leaked, _start, _end)
_bad = span_scan.find_tokens_outside_spans(_doc_leaked, "29.10", [_span_leaked])
check(len(_bad) == 1,
      "campaign token leaked into primary-result prose outside its passage is caught")
check(span_scan.count_tokens_in_spans(_doc_leaked, "29.10", [_span_leaked]) == 1,
      "the in-passage occurrence is still counted correctly alongside the stray one")

print("RESULT:", "ALL TESTS PASSED" if failures == 0 else f"{failures} FAILED")
sys.exit(1 if failures else 0)
