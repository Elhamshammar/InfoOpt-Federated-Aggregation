#!/usr/bin/env python3
"""Generate the manuscript and supplement tables from the verified reanalysis outputs.

Inputs (data/, copied unchanged from the verified figure package
assets/ITSL_verified_figures/source_data/), plus the CIFAR-100 FedOpt
sensitivity-only output read directly from the frozen, checksum-verified
assets/ITSL_reanalysis_sensitivity_v2/outputs/sensitivity_cifar_fedopt.json:
    primary_summary.csv           primary retrospective summaries (one row per dataset x arm)
    paired_contrasts.json         six-contrast primary family (Holm m = 6)
    sensitivity_latest_repeat.json  latest-repeat sensitivity (FedAvg / InfoOpt-Full only)
    ra_strata_summaries.json      RA-InfoOpt per-stratum summaries

Outputs (tables/):
    table3_retrospective.tex      main Table 3
    table4_primary_contrasts.tex  main Table 4
    tableA_primary.tex            supplement: primary family with sd of differences and d_z
    tableA_sensitivity.tex        supplement: latest-repeat sensitivity (summaries + contrasts)
    tableA_ra_strata.tex          supplement: RA-InfoOpt strata (descriptive only)

Design rule (fix for the earlier draft): PRIMARY and SENSITIVITY records are held in
separate containers and are never merged. A primary cell's status (reported /
withheld / excluded / flagged) is decided only from the primary record plus the
explicit STATUS rules below; no sensitivity record can overwrite or fill a primary
cell. Every inconsistency raises an error instead of being silently resolved.

Usage: python3 scripts/make_tables.py   (from any directory)
"""
import csv
import json
import math
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "tables")

DATASETS = ["cifar100", "eurosat", "pathmnist"]
DS_LABEL = {"cifar100": "CIFAR-100", "eurosat": "EuroSAT", "pathmnist": "PathMNIST"}
ARMS = [  # (csv row key, LaTeX label) in display order
    ("fedavg", "FedAvg"),
    ("gamma0.5", "FedAvg-$\\gamma_{\\mathrm{damp}}=0.50$"),
    ("gamma0.75", "FedAvg-$\\gamma_{\\mathrm{damp}}=0.75$"),
    ("InfoOpt_Full", "InfoOpt-Full (AIB)"),
    ("RA_InfoOpt", "RA-InfoOpt (AIB)"),
    ("fedprox", "FedProx"),
    ("fednova", "FedNova"),
    ("fedopt", "FedOpt"),
    ("ibfl", "IB-FL"),
    ("feddp", "FedDP"),
    ("feddyn", "FedDyn"),
    ("scaffold", "SCAFFOLD"),
]

# Explicit status rules for the PRIMARY table (documented in the manuscript, Sec. 4.5).
WITHHELD_ROWS = {"RA_InfoOpt"}                       # configuration strata differ across seeds
EXCLUDED_PRIMARY = {("cifar100", "fedopt")}          # strict configured-early-stop rule
FLAGGED_AUDIT_ROWS = {"feddyn", "scaffold"}          # implementation-provenance audit observations


@dataclass(frozen=True)
class PrimaryRecord:
    dataset: str
    row: str
    n: int
    mean: float | None
    sd: float | None
    seeds: str
    note: str


@dataclass(frozen=True)
class SensitivityRecord:
    dataset: str
    row: str
    n: int
    primary_mean: float
    sensitivity_mean: float
    delta: float


@dataclass(frozen=True)
class ExcludedCellSensitivity:
    """Sensitivity-only value for a cell EXCLUDED from the primary summary.

    Held in its own container; it can only ever be printed in the Table 3 footnote.
    """
    dataset: str
    row: str
    n: int
    mean: float
    sd: float
    provenance: str


SENS_PKG = os.path.join(ROOT, "assets", "ITSL_reanalysis_sensitivity_v2")
SENS_PKG_ZIP_SHA256 = "ba5f20e430b6eafb551b4ed6c95240c4797b11141fa3ccc984fa0fa482469a96"  # as recorded in the figure package README

# Four-arm controlled experiment (revision 5): frozen read-only package, verified
# against its own SHA256SUMS.txt, kept separate from the historical data/ files.
FOURARM_PKG = os.path.join(ROOT, "assets", "ITSL_four_arm_20261010")
FOURARM_WORK = os.path.join(ROOT, "data_fourarm")
FOURARM_ARMS = ["fedavg_aiboff", "fedavg_aibon", "infoopt_aiboff", "infoopt_aibon"]
FOURARM_ARM_LABEL = {"fedavg_aiboff": "FedAvg", "fedavg_aibon": "FedAvg + AIB",
                      "infoopt_aiboff": "InfoOpt-Full $-$ AIB", "infoopt_aibon": "InfoOpt-Full (AIB)"}

# Matched RA-InfoOpt experiment (revision 7): frozen read-only package, verified
# against its own SHA256SUMS.txt, kept separate from both the historical data/
# files and the revision-5 four-arm package/data_fourarm/ working copies.
RA_PKG = os.path.join(ROOT, "assets", "ITSL_RA_matched_20261010")
RA_WORK = os.path.join(ROOT, "data_ra")
RA_ARMS = ["fedavg_aibon", "infoopt_aibon", "ra_probeoff", "ra_probeon"]
RA_ARM_LABEL = {"fedavg_aibon": "FedAvg (AIB)", "infoopt_aibon": "InfoOpt-Full (AIB)",
                "ra_probeoff": "RA-InfoOpt, probe off (AIB)", "ra_probeon": "RA-InfoOpt, probe on (AIB)"}

# Saved-checkpoint CIFAR-100 step-size pilot (revision 8): frozen read-only evidence
# package, verified against its own SHA256SUMS.txt, kept separate from every other
# frozen package and every other data_* working-copy directory. One-step evaluations
# from saved checkpoints (rounds 5/50/100), one partition (seed 42), dev training
# seed 201, held-out training seeds 202/203; repetitions within a (cohort, seed,
# round) cell are clustered, not independent training seeds.
STEPSIZE_PKG = os.path.join(ROOT, "assets", "ITSL_stepsize_pilot_20261011")
STEPSIZE_WORK = os.path.join(ROOT, "data_stepsize")
STEPSIZE_HELD_SEEDS = [202, 203]
STEPSIZE_CONTROLS = ["global", "phase"]
STEPSIZE_CONTROL_LABEL = {"global": "global fixed ($\\gamma=0.5$)",
                           "phase": "phase-specific fixed (0.75/0.5/0.5)"}
# Frozen verdict strings (GO_NO_GO.md v3): any deviation is a protocol change and
# must not be silently absorbed into the manuscript table.
STEPSIZE_EXPECTED_VERDICTS = {
    "adaptive_headroom": "INSUFFICIENT",
    "claude_comparator": "NO-GO",
    "exploratory_variability_signal": "EXPLORATORY_PROBE",
    "overall_recommendation": "ADAPTIVE_BENEFIT_NOT_ESTABLISHED_IN_THIS_PILOT",
}


def _verify_pkg_file(pkg, rel):
    """Check one file of a frozen package against that package's SHA256SUMS.txt."""
    import hashlib
    sums = {}
    for line in open(os.path.join(pkg, "SHA256SUMS.txt")):
        h, name = line.strip().split(maxsplit=1)
        sums[name.lstrip("*").replace("\\", "/")] = h
    if rel not in sums:
        raise ValueError(f"{rel} not listed in {pkg}/SHA256SUMS.txt")
    got = hashlib.sha256(open(os.path.join(pkg, rel), "rb").read()).hexdigest()
    if got != sums[rel]:
        raise ValueError(f"checksum mismatch for {rel}: {got} != {sums[rel]}")
    return got


def load_excluded_cell_sensitivity(pkg=SENS_PKG):
    """Read outputs/sensitivity_cifar_fedopt.json from the frozen sensitivity v2 package.

    The file's own schema is {label, n, mean, sd, runs:[{seed, run, round100_acc_pct}]}.
    The cell identity (dataset, row, role) is taken from the frozen source inventory,
    matched by run name, never assumed. Mean and SD are recomputed from the runs.
    """
    import statistics
    out_rel = "outputs/sensitivity_cifar_fedopt.json"
    inv_rel = "frozen_inputs/sensitivity_source_inventory_FROZEN_20261008.csv"
    if not os.path.exists(os.path.join(pkg, out_rel)):
        return []
    _verify_pkg_file(pkg, out_rel)
    _verify_pkg_file(pkg, inv_rel)
    js = json.load(open(os.path.join(pkg, out_rel)))
    if "SENSITIVITY ONLY" not in js.get("label", ""):
        raise ValueError(f"{out_rel}: label does not declare SENSITIVITY ONLY")
    inv = {r["run_name"]: r for r in csv.DictReader(open(os.path.join(pkg, inv_rel)))}
    idents = set()
    for r in js["runs"]:
        rec = inv.get(r["run"])
        if rec is None:
            raise ValueError(f"run {r['run']} not in the frozen inventory")
        if int(rec["seed"]) != int(r["seed"]):
            raise ValueError(f"seed mismatch for {r['run']}")
        idents.add((rec["dataset"], rec["manuscript_row"], rec["role"]))
    if len(idents) != 1:
        raise ValueError(f"runs map to more than one cell/role: {idents}")
    dataset, row, role = idents.pop()
    if role != "cifar_fedopt_sensitivity_only":
        raise ValueError(f"unexpected inventory role {role}")
    vals = [float(r["round100_acc_pct"]) for r in js["runs"]]
    if len(vals) != int(js["n"]) or len({r["seed"] for r in js["runs"]}) != len(vals):
        raise ValueError("n does not match the number of distinct-seed runs")
    if abs(statistics.mean(vals) - js["mean"]) > 1e-9 or abs(statistics.stdev(vals) - js["sd"]) > 1e-9:
        raise ValueError("reported mean/sd do not match the per-seed values")
    rec = ExcludedCellSensitivity(dataset, row, int(js["n"]), float(js["mean"]), float(js["sd"]),
                                  f"{os.path.relpath(os.path.join(pkg, out_rel), ROOT)} (checksum-verified)")
    if (rec.dataset, rec.row) not in EXCLUDED_PRIMARY:
        raise ValueError(f"sensitivity-only value supplied for a non-excluded cell {(rec.dataset, rec.row)}")
    return [rec]


def verify_fourarm_package(pkg=FOURARM_PKG):
    """Verify every file of the frozen four-arm package against its own
    SHA256SUMS.txt, then verify that each working copy in data_fourarm/ is
    byte-identical to the frozen copy. Raises on any mismatch."""
    import hashlib
    sums = {}
    for line in open(os.path.join(pkg, "SHA256SUMS.txt")):
        h, name = line.strip().split(maxsplit=1)
        sums[name.lstrip("*")] = h
    for rel, h in sums.items():
        got = hashlib.sha256(open(os.path.join(pkg, rel), "rb").read()).hexdigest()
        if got != h:
            raise ValueError(f"four-arm package checksum mismatch for {rel}: {got} != {h}")
    work_map = {
        "primary_summary.csv": "fourarm_primary_summary.csv",
        "primary_contrasts.json": "fourarm_primary_contrasts.json",
        "aib_interactions.json": "fourarm_aib_interactions.json",
        "per_seed_round100.csv": "fourarm_per_seed_round100.csv",
        "verification_report.json": "fourarm_verification_report.json",
    }
    for frozen_name, work_name in work_map.items():
        a = open(os.path.join(pkg, frozen_name), "rb").read()
        b = open(os.path.join(FOURARM_WORK, work_name), "rb").read()
        if a != b:
            raise ValueError(f"data_fourarm/{work_name} is not byte-identical to the frozen package copy")
    return sums


def load_fourarm_primary():
    """12 rows: four arms x three datasets. Independently checked against
    per_seed_round100.csv by scripts/check_numbers.py, not here."""
    recs = {}
    with open(os.path.join(FOURARM_WORK, "fourarm_primary_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            key = (r["dataset"], r["arm"])
            if key in recs:
                raise ValueError(f"duplicate four-arm primary record {key}")
            recs[key] = {"dataset": r["dataset"], "arm": r["arm"], "n": int(r["n"]),
                         "mean": float(r["mean_accuracy_percent"]), "sd": float(r["sd_accuracy_pp"])}
    if len(recs) != 12:
        raise ValueError(f"four-arm primary summary must contain 12 rows (4 arms x 3 datasets), found {len(recs)}")
    for d in DATASETS:
        for a in FOURARM_ARMS:
            if (d, a) not in recs:
                raise ValueError(f"missing four-arm primary record {(d, a)}")
    return recs


def load_fourarm_contrasts():
    """Six within-AIB InfoOpt-Full - FedAvg contrasts: {aiboff, aibon} x three datasets."""
    cs = json.load(open(os.path.join(FOURARM_WORK, "fourarm_primary_contrasts.json")))
    if len(cs) != 6:
        raise ValueError(f"four-arm primary contrast family must contain six contrasts, found {len(cs)}")
    seen = set()
    for c in cs:
        key = (c["dataset"], c["aib"])
        if key in seen:
            raise ValueError(f"duplicate four-arm contrast {key}")
        seen.add(key)
    for d in DATASETS:
        for aib in ["aiboff", "aibon"]:
            if (d, aib) not in seen:
                raise ValueError(f"missing four-arm contrast {(d, aib)}")
    return cs


def load_fourarm_interactions():
    """Three aggregation x AIB interaction contrasts, one per dataset."""
    ints = json.load(open(os.path.join(FOURARM_WORK, "fourarm_aib_interactions.json")))
    if len(ints) != 3:
        raise ValueError(f"four-arm interaction family must contain three contrasts, found {len(ints)}")
    seen = {c["dataset"] for c in ints}
    if seen != set(DATASETS):
        raise ValueError(f"four-arm interactions must cover exactly the three datasets, found {seen}")
    return ints


def verify_ra_package(pkg=RA_PKG):
    """Verify every file of the frozen matched RA-InfoOpt package against its own
    SHA256SUMS.txt, then verify that each working copy in data_ra/ is byte-identical
    to the frozen copy. Raises on any mismatch. Does not touch assets/ITSL_four_arm_20261010/
    or data_fourarm/, which remain the revision-5 four-arm evidence."""
    import hashlib
    sums = {}
    for line in open(os.path.join(pkg, "SHA256SUMS.txt")):
        h, name = line.strip().split(maxsplit=1)
        sums[name.lstrip("*")] = h
    for rel, h in sums.items():
        got = hashlib.sha256(open(os.path.join(pkg, rel), "rb").read()).hexdigest()
        if got != h:
            raise ValueError(f"RA-matched package checksum mismatch for {rel}: {got} != {h}")
    work_map = {
        "primary_summary.csv": "ra_primary_summary.csv",
        "primary_contrasts.json": "ra_primary_contrasts.json",
        "probe_contrasts.json": "ra_probe_contrasts.json",
        "per_seed_round100.csv": "ra_per_seed_round100.csv",
        "verification_report.json": "ra_verification_report.json",
    }
    for frozen_name, work_name in work_map.items():
        a = open(os.path.join(pkg, frozen_name), "rb").read()
        b = open(os.path.join(RA_WORK, work_name), "rb").read()
        if a != b:
            raise ValueError(f"data_ra/{work_name} is not byte-identical to the frozen package copy")
    return sums


def load_ra_primary():
    """12 rows: four arms (all AIB-on) x three datasets."""
    recs = {}
    with open(os.path.join(RA_WORK, "ra_primary_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            key = (r["dataset"], r["arm"])
            if key in recs:
                raise ValueError(f"duplicate RA-matched primary record {key}")
            recs[key] = {"dataset": r["dataset"], "arm": r["arm"], "n": int(r["n"]),
                         "mean": float(r["mean_percent"]), "sd": float(r["sd_percent"])}
    if len(recs) != 12:
        raise ValueError(f"RA-matched primary summary must contain 12 rows (4 arms x 3 datasets), found {len(recs)}")
    for d in DATASETS:
        for a in RA_ARMS:
            if (d, a) not in recs:
                raise ValueError(f"missing RA-matched primary record {(d, a)}")
    return recs


def load_ra_contrasts():
    """Six primary contrasts: {ra_probeoff - fedavg_aibon, ra_probeoff - infoopt_aibon} x three datasets.
    Holm family size six, separate from the four-arm and retrospective families."""
    cs = json.load(open(os.path.join(RA_WORK, "ra_primary_contrasts.json")))
    if len(cs) != 6:
        raise ValueError(f"RA-matched primary contrast family must contain six contrasts, found {len(cs)}")
    seen = set()
    for c in cs:
        key = (c["dataset"], c["contrast"])
        if key in seen:
            raise ValueError(f"duplicate RA-matched contrast {key}")
        seen.add(key)
    for d in DATASETS:
        for base in ["ra_probeoff minus fedavg_aibon", "ra_probeoff minus infoopt_aibon"]:
            if (d, base) not in seen:
                raise ValueError(f"missing RA-matched contrast {(d, base)}")
    return cs


def load_ra_probe_contrasts():
    """Three secondary (instrumentation-sensitivity) contrasts: ra_probeon - ra_probeoff,
    one per dataset. Separate Holm family of three; not pooled with the primary family."""
    cs = json.load(open(os.path.join(RA_WORK, "ra_probe_contrasts.json")))
    if len(cs) != 3:
        raise ValueError(f"RA-matched probe contrast family must contain three contrasts, found {len(cs)}")
    seen = {c["dataset"] for c in cs}
    if seen != set(DATASETS):
        raise ValueError(f"RA-matched probe contrasts must cover exactly the three datasets, found {seen}")
    for c in cs:
        if c["contrast"] != "ra_probeon minus ra_probeoff":
            raise ValueError(f"unexpected RA-matched probe contrast label {c['contrast']!r}")
    return cs


def verify_stepsize_package(pkg=STEPSIZE_PKG):
    """Verify every file of the frozen saved-checkpoint step-size pilot package
    against its own SHA256SUMS.txt, then verify that each working copy in
    data_stepsize/ is byte-identical to the frozen copy. Raises on any mismatch.
    Does not touch any other assets/ or data_*/ directory."""
    import hashlib
    sums = {}
    for line in open(os.path.join(pkg, "SHA256SUMS.txt")):
        h, name = line.strip().split(maxsplit=1)
        sums[name.lstrip("*")] = h
    for rel, h in sums.items():
        p = os.path.join(pkg, rel)
        if not os.path.exists(p):
            print(f"WARNING: step-size pilot package file missing: {rel} "
                  f"(listed in SHA256SUMS.txt but not present); skipping checksum")
            continue
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != h:
            raise ValueError(f"step-size pilot package checksum mismatch for {rel}: {got} != {h}")
    work_map = {
        "pilot/results/pilot_decision.json": "stepsize_pilot_decision.json",
        "corrected_exploratory_v2/exploratory_signal_held_out_corrected.json":
            "stepsize_exploratory_held_out_corrected.json",
        "corrected_exploratory_v2/exploratory_signal_dev_corrected.json":
            "stepsize_exploratory_dev_corrected.json",
        "buffer_policy_v2/cohort_summary.csv": "stepsize_buffer_cohort_summary.csv",
        "buffer_policy_v2/report.md": "stepsize_buffer_report.md",
        "per_example_correction/CORRECTION_REPORT.md": "stepsize_correction_report.md",
    }
    for frozen_name, work_name in work_map.items():
        a = open(os.path.join(pkg, frozen_name), "rb").read()
        b = open(os.path.join(STEPSIZE_WORK, work_name), "rb").read()
        if a != b:
            raise ValueError(f"data_stepsize/{work_name} is not byte-identical to the frozen package copy")
    return sums


def load_stepsize_decision():
    """Load pilot_decision.json (frozen decision record). Asserts the four
    top-level verdict strings match the frozen GO_NO_GO.md v3 vocabulary before
    extracting any per-seed G-value, so a silent protocol change cannot flow
    into the manuscript table unnoticed. Returns the per-seed-per-control mean G
    for adaptive headroom (validation-selected scaling) and for the claude
    raw-shrinkage comparator, against both the global and the phase-specific
    fixed control, for both held-out seeds."""
    d = json.load(open(os.path.join(STEPSIZE_WORK, "stepsize_pilot_decision.json")))
    statuses = d["statuses"]
    for key, expected in STEPSIZE_EXPECTED_VERDICTS.items():
        if statuses.get(key) != expected:
            raise ValueError(f"stepsize pilot_decision.json[{key}] = {statuses.get(key)!r}, "
                              f"expected frozen verdict {expected!r} (GO_NO_GO.md v3)")
    headroom = d["details"]["adaptive_headroom"]["per_seed_per_control"]
    claude = d["details"]["claude_comparator"]["per_seed_per_control"]
    out = {"headroom": {}, "claude": {}}
    for seed in STEPSIZE_HELD_SEEDS:
        for control, headroom_key, claude_key in [("global", "global", "vs_global_fixed"),
                                                    ("phase", "phase", "vs_phase_fixed")]:
            out["headroom"][(seed, control)] = headroom[str(seed)][headroom_key]["mean_overall_G"]
            out["claude"][(seed, control)] = claude[claude_key]["per_seed"][str(seed)]["mean_G"]
    out["headroom_gate_failure_reason"] = d["details"]["adaptive_headroom"]["gate_failure_reason"]
    out["claude_gate_failure_reason"] = d["details"]["claude_comparator"]["gate_failure_reason"]
    return out


def load_stepsize_exploratory():
    """Load the CORRECTED (v2) exploratory held-out predictor comparison. The
    corrected files, not pilot_decision.json's own stale exploratory_variability_signal
    block (fit_status='no_cells' there; see PILOT_CLOSURE.md Sec. 2), are the
    authoritative source. Asserts fit_status == 'fit' for both predictors."""
    d = json.load(open(os.path.join(STEPSIZE_WORK, "stepsize_exploratory_held_out_corrected.json")))
    if d.get("tag") != "held_out":
        raise ValueError(f"unexpected exploratory tag {d.get('tag')!r}")
    out = {}
    for p in d["predictors"]:
        if p["fit_status"] != "fit":
            raise ValueError(f"exploratory predictor {p['predictor_id']} fit_status "
                              f"{p['fit_status']!r} != 'fit'; corrected file should have no_cells repaired")
        out[p["predictor_id"]] = {"mae_held": p["mae_held"], "mse_held": p["mse_held"],
                                   "n_dev_cells_used": p["n_dev_cells_used"],
                                   "n_held_cells_used": p["n_held_cells_used"]}
    for pid in ["phase_only", "phase_plus_variability"]:
        if pid not in out:
            raise ValueError(f"missing corrected exploratory predictor {pid}")
    return out


def load_stepsize_buffer():
    """Independently recompute the buffer-policy cohort-mean comparisons from
    buffer_policy_v2/cohort_summary.csv (raw per-cohort rows), rather than
    transcribing buffer_policy_v2/report.md's own numbers. Held-out = seeds
    202/203; dev = seed 201 (selection source, not held-out). Trajectory rows
    (is_trajectory=True) are excluded from the cohort-mean, matching the
    frozen protocol's trajectory-separate-reporting rule."""
    rows = list(csv.DictReader(open(os.path.join(STEPSIZE_WORK, "stepsize_buffer_cohort_summary.csv"))))
    comparisons = ["A_freeze_buffers_at_gamma1", "B_valsel_vs_full_gamma1",
                   "B_valsel_vs_params_only_gamma1", "C_valsel_vs_fixed_global", "C_valsel_vs_fixed_phase"]
    out = {}
    for comp in comparisons:
        held = [float(r["mean_loss_improvement"]) for r in rows
                if r["comparison"] == comp and int(r["seed"]) in (202, 203) and r["is_trajectory"] != "True"]
        dev = [float(r["mean_loss_improvement"]) for r in rows
               if r["comparison"] == comp and int(r["seed"]) == 201 and r["is_trajectory"] != "True"]
        if len(held) != 18 or len(dev) != 9:
            raise ValueError(f"stepsize buffer comparison {comp}: expected 18 held-out / 9 dev "
                              f"cohort-mean rows, found {len(held)} / {len(dev)}")
        out[comp] = {"held_mean": sum(held) / len(held), "held_n": len(held),
                      "dev_mean": sum(dev) / len(dev), "dev_n": len(dev)}
    return out


# ------------------------------------------------------------------- Table A.6 (new, revision 8, supplement)
def tableA6_stepsize_pilot(decision):
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data_stepsize/stepsize_pilot_decision.json --",
        "% do not edit by hand",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Saved-checkpoint CIFAR-100 step-size pilot (new; one partition, seed 42; SmallCNN,",
        "AIB disabled; development training seed 201; held-out training seeds 202 and 203; one-step",
        "evaluations from checkpoints saved at rounds 5, 50, and 100 -- not end-to-end adaptive-controller",
        "training; repetitions within a cohort/seed/checkpoint are clustered, not independent training",
        "seeds). $G = L_{\\mathrm{fixed\\ control}} - L_{\\mathrm{candidate}}$ (positive = candidate improves",
        "on the fixed control), averaged over checkpoint rounds per the frozen protocol. The headroom gate",
        "(both held-out seeds, both controls: mean $G\\geq+0.01$, $\\geq2$ of 3 phases positive, worst-phase",
        "deterioration $\\leq0.02$) is NOT met by validation-selected scaling; the tested raw-shrinkage rule",
        "is worse than both fixed controls on every checkpoint of both held-out seeds.}",
        "\\label{app:tab:stepsize_pilot}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\small",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "\\textbf{Held-out seed} & \\textbf{Fixed control} & \\textbf{Validation-selected $G$} & "
        "\\textbf{Shrinkage-rule $G$} \\\\",
        "\\midrule",
    ]
    for seed in STEPSIZE_HELD_SEEDS:
        for j, control in enumerate(STEPSIZE_CONTROLS):
            lab = str(seed) if j == 0 else ""
            lines.append(f"{lab} & {STEPSIZE_CONTROL_LABEL[control]} & "
                         f"{signed(decision['headroom'][(seed, control)], 6)} & "
                         f"{signed(decision['claude'][(seed, control)], 6)} \\\\")
        lines.append("\\midrule" if seed != STEPSIZE_HELD_SEEDS[-1] else "\\bottomrule")
    lines += [
        "\\end{tabular}",
        "\\par\\vspace{3pt}\\begin{minipage}{0.92\\textwidth}\\footnotesize",
        "Both named headroom outcomes fail the frozen gate on at least one held-out seed against each",
        "fixed control (seed 203: fewer than 2 of 3 phases positive against either control). The",
        "shrinkage rule fails the frozen mean-improvement screen against both controls on both held-out",
        "seeds and performs worse at every evaluated checkpoint. Neither result licenses a claim that",
        "step-size control is ineffective in general; both are",
        "measured negatives for this bounded pilot only (Sec.~\\ref{app:stepsize_pilot}).",
        "\\end{minipage}",
        "\\end{table}",
    ]
    write("tableA6_stepsize_pilot.tex", lines)


def load_primary():
    recs = {}
    with open(os.path.join(DATA, "primary_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            key = (r["dataset"], r["row"])
            if key in recs:
                raise ValueError(f"duplicate primary record {key}")
            n = int(r["n"])
            recs[key] = PrimaryRecord(r["dataset"], r["row"], n,
                                      float(r["mean"]) if r["mean"] else None,
                                      float(r["sd"]) if r["sd"] else None,
                                      r["seeds"], r["note"])
    return recs


def load_sensitivity():
    js = json.load(open(os.path.join(DATA, "sensitivity_latest_repeat.json")))
    summaries, contrasts = [], []
    for r in js["results"]:
        if "-" in r["row"]:
            contrasts.append(r)
        else:
            summaries.append(SensitivityRecord(r["dataset"], r["row"], int(r["n"]),
                                               float(r["primary_mean_earliest"]),
                                               float(r["sensitivity_mean_latest"]),
                                               float(r["delta"])))
    return js["label"], summaries, contrasts


def primary_status(rec: PrimaryRecord) -> str:
    key = (rec.dataset, rec.row)
    if rec.row in WITHHELD_ROWS:
        status = "withheld"
    elif key in EXCLUDED_PRIMARY:
        status = "excluded"
    elif rec.row in FLAGGED_AUDIT_ROWS:
        status = "flagged"
    else:
        status = "reported"
    has_value = rec.n > 0 and rec.mean is not None
    if status in ("withheld", "excluded") and has_value:
        raise ValueError(f"{key}: rule says {status} but primary output has a value")
    if status in ("reported", "flagged") and not has_value:
        raise ValueError(f"{key}: rule says {status} but primary output has no value ({rec.note})")
    return status


def signed(x, nd=3):
    s = f"{x:+.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote tables/" + name)


# ----------------------------------------------------------------------------- Table 3
def table3(primary, excl_sens):
    for key in [(d, a) for d in DATASETS for a, _ in ARMS]:
        if key not in primary:
            raise ValueError(f"missing primary record {key}")
    # Sensitivity values for EXCLUDED primary cells come only from the SEPARATE container.
    sens_index = {}
    for s in excl_sens:
        if (s.dataset, s.row) in sens_index:
            raise ValueError(f"duplicate sensitivity-only record {(s.dataset, s.row)}")
        sens_index[(s.dataset, s.row)] = s
    excl_notes = []
    for d, a in sorted(EXCLUDED_PRIMARY):
        s = sens_index.get((d, a))
        if s is None:
            excl_notes.append(f"{DS_LABEL[d]} FedOpt: excluded from the primary summary under the strict "
                              "configured-early-stop rule; the supplied sensitivity outputs contain no value for "
                              "this cell, so none is reported.")
        else:
            excl_notes.append(f"{DS_LABEL[d]} FedOpt: excluded from the primary summary under the strict "
                              "configured-early-stop rule. Sensitivity value only, not used in any comparison or "
                              f"ranking: {s.mean:.2f} $\\pm$ {s.sd:.2f} ({s.n}).")
    body = []
    for a, lab in ARMS:
        cells = []
        for d in DATASETS:
            rec = primary[(d, a)]
            st = primary_status(rec)
            if st == "withheld":
                cells.append("\\emph{withheld}")
            elif st == "excluded":
                cells.append("\\emph{excluded}$^{\\ast}$")
            else:
                dag = "$^{\\dagger}$" if st == "flagged" else ""
                cells.append(f"{rec.mean:.2f} $\\pm$ {rec.sd:.2f} ({rec.n}){dag}")
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if a in ("gamma0.75", "RA_InfoOpt"):
            body.append("\\addlinespace[2pt]")
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data/primary_summary.csv -- do not edit by hand",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Retrospective reanalysis: terminal test accuracy (\\%) at round~100 under the canonical",
        "setting ($\\alpha_{\\mathrm{Dir}}=0.3$), reported as mean $\\pm$ sample SD (number of retained runs).",
        "Runs were selected under the frozen retrospective rule (Section~\\ref{subsec:provenance}); these",
        "summaries are not the unrecovered historical selection behind the originally printed table, and a",
        "match with a printed value does not certify a run as its historical source. Combined RA-InfoOpt",
        "estimates are withheld because configuration strata differ across seeds (per-stratum values in",
        "Supplementary Table~A.3). Only the two AIB-labelled arms include AIB, so differences between them",
        "and the other arms combine architecture and aggregation. No method ranking is implied and no",
        "maximum is highlighted.}",
        "\\label{tab:canonical}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Arm} & \\textbf{CIFAR-100} & \\textbf{EuroSAT} & \\textbf{PathMNIST} \\\\",
        "\\midrule",
    ] + body + [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{3pt}\\begin{minipage}{0.92\\textwidth}\\footnotesize",
        "$^{\\ast}$" + " ".join(excl_notes),
        "$^{\\dagger}$FedDyn and SCAFFOLD values are flagged implementation-provenance audit observations;",
        "they are excluded from method rankings and from conclusions about method performance, and the",
        "flag does not by itself establish defective code or code-version drift.",
        "\\end{minipage}",
        "\\end{table*}",
    ]
    write("table3_retrospective.tex", lines)


# ----------------------------------------------------------------------------- Table 4 + A.1
def load_contrasts():
    js = json.load(open(os.path.join(DATA, "paired_contrasts.json")))
    cs = js["contrasts"]
    if len(cs) != 6:
        raise ValueError("primary family must contain six contrasts")
    return cs


def table4(cs):
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data/paired_contrasts.json -- do not edit by hand",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Retrospective reanalysis: primary matched-seed contrasts against FedAvg (terminal test",
        "accuracy, canonical $\\alpha_{\\mathrm{Dir}}=0.3$). $\\Delta$ is method minus FedAvg in percentage",
        "points (pp); positive values favor the adaptive arm. 95\\% CIs are paired-$t$ intervals. The",
        "Holm--Bonferroni family retains all six contrasts; the three RA-InfoOpt contrasts are not estimable",
        "because configuration strata differ across seeds and are counted in the family size without an",
        "estimate. $d_z=\\bar d/s_d$ is computed from the verified mean and SD of the paired differences.",
        "Intervals that include zero indicate uncertainty; they do not establish equivalence or",
        "ineffectiveness. Each contrast compares an AIB arm with a non-AIB arm and therefore does not",
        "isolate the aggregation rule (Section~\\ref{subsec:provenance}).}",
        "\\label{tab:paired}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Contrast} & $n$ & $\\Delta$ \\textbf{(pp)} & \\textbf{95\\% CI} & "
        "$p_{\\mathrm{raw}}$ & $p_{\\mathrm{Holm}}$ & $d_z$ \\\\",
        "\\midrule",
    ]
    for i, d in enumerate(DATASETS):
        sub = [c for c in cs if c["dataset"] == d]
        sub.sort(key=lambda c: 0 if c["contrast"].startswith("InfoOpt") else 1)
        for j, c in enumerate(sub):
            lab = DS_LABEL[d] if j == 0 else ""
            name = "InfoOpt-Full$-$FedAvg" if c["contrast"].startswith("InfoOpt") else "RA-InfoOpt$-$FedAvg"
            if c["estimable"]:
                dz = c["mean_diff_pp"] / c["sd_diff"]
                lo, hi = c["ci95"]
                lines.append(f"{lab} & {name} & {c['n']} & {signed(c['mean_diff_pp'])} & "
                             f"[{signed(lo)}, {signed(hi)}] & {c['p_raw']:.3f} & {c['p_holm']:.3f} & {signed(dz, 2)} \\\\")
            else:
                lines.append(f"{lab} & {name} & \\multicolumn{{6}}{{c}}{{\\emph{{not estimable: configuration strata differ across seeds}}}} \\\\")
        if i < 2:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write("table4_primary_contrasts.tex", lines)


def tableA_primary(cs):
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data/paired_contrasts.json -- do not edit by hand",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Retrospective primary family (verified output). $s_d$: sample SD of the five paired",
        "differences; $d_z=\\bar d/s_d$. Holm adjustment uses family size six. RA-InfoOpt contrasts are not",
        "estimable because configuration strata differ across seeds. Values supersede the historical",
        "Table~A.1 of the earlier supplement.}",
        "\\label{app:tab:primary_holm}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Contrast} & $n$ & $\\Delta$ (pp) & $s_d$ & 95\\% CI & $p_{\\mathrm{raw}}$ & "
        "$p_{\\mathrm{Holm}}$ & $d_z$ \\\\",
        "\\midrule",
    ]
    for i, d in enumerate(DATASETS):
        sub = sorted([c for c in cs if c["dataset"] == d], key=lambda c: 0 if c["contrast"].startswith("InfoOpt") else 1)
        for j, c in enumerate(sub):
            lab = DS_LABEL[d] if j == 0 else ""
            name = "InfoOpt-Full$-$FedAvg" if c["contrast"].startswith("InfoOpt") else "RA-InfoOpt$-$FedAvg"
            if c["estimable"]:
                lo, hi = c["ci95"]
                lines.append(f"{lab} & {name} & {c['n']} & {signed(c['mean_diff_pp'])} & {c['sd_diff']:.3f} & "
                             f"[{signed(lo)}, {signed(hi)}] & {c['p_raw']:.3f} & {c['p_holm']:.3f} & "
                             f"{signed(c['mean_diff_pp'] / c['sd_diff'], 2)} \\\\")
            else:
                lines.append(f"{lab} & {name} & \\multicolumn{{7}}{{c}}{{\\emph{{not estimable}}}} \\\\")
        if i < 2:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tableA_primary.tex", lines)


# ----------------------------------------------------------------------------- A: sensitivity
def tableA_sensitivity(cs, sens_summ, sens_contr):
    prim = {c["dataset"]: c for c in cs if c["estimable"]}
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data/sensitivity_latest_repeat.json and",
        "% data/paired_contrasts.json -- do not edit by hand",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Repeat-selection sensitivity for FedAvg and InfoOpt-Full only. The primary rule retains",
        "the earliest eligible run per seed within a configuration stratum; the sensitivity rule retains the",
        "latest. Cells with a single eligible run reuse that run. (a) Summary means; (b) InfoOpt-Full$-$FedAvg",
        "contrasts under each rule. Sensitivity contrasts are not part of the Holm family and carry raw",
        "$p$-values only. No other arm was assessed.}",
        "\\label{app:tab:sensitivity}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\multicolumn{5}{l}{\\emph{(a) Summary means (\\%), $n=5$ each}}\\\\",
        "\\textbf{Dataset} & \\textbf{Arm} & \\textbf{Earliest (primary)} & \\textbf{Latest} & \\textbf{Change (pp)} \\\\",
        "\\midrule",
    ]
    for d in DATASETS:
        for j, s in enumerate([x for x in sens_summ if x.dataset == d]):
            lab = DS_LABEL[d] if j == 0 else ""
            arm = "FedAvg" if s.row == "fedavg" else "InfoOpt-Full"
            lines.append(f"{lab} & {arm} & {s.primary_mean:.3f} & {s.sensitivity_mean:.3f} & {signed(s.delta)} \\\\")
    lines += [
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{(b) InfoOpt-Full$-$FedAvg contrast (pp), 95\\% paired-$t$ CI, raw $p$}}\\\\",
        "\\textbf{Dataset} & & \\textbf{Primary (earliest)} & \\textbf{Sensitivity (latest)} & \\\\",
        "\\midrule",
    ]
    for d in DATASETS:
        p = prim[d]
        s = [x for x in sens_contr if x["dataset"] == d][0]
        lines.append(f"{DS_LABEL[d]} & & {signed(p['mean_diff_pp'])} [{signed(p['ci95'][0])}, {signed(p['ci95'][1])}]; "
                     f"$p={p['p_raw']:.3f}$ & {signed(s['sensitivity_mean_latest_diff'])} "
                     f"[{signed(s['ci95'][0])}, {signed(s['ci95'][1])}]; $p={s['p_raw']:.3f}$ & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tableA_sensitivity.tex", lines)


def tableA_strata():
    js = json.load(open(os.path.join(DATA, "ra_strata_summaries.json")))
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data/ra_strata_summaries.json -- do not edit by hand",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{RA-InfoOpt configuration strata (descriptive only). One run per seed per stratum (earliest",
        "within stratum). Strata are identified by configuration-hash prefix. No combined five-seed",
        "estimate is formed and no comparison across strata or with FedAvg is made. Seed~100 appears in two",
        "strata for every dataset.}",
        "\\label{app:tab:ra_strata}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Stratum} & \\textbf{Seeds} & $n$ & \\textbf{Mean $\\pm$ SD (\\%)} \\\\",
        "\\midrule",
    ]
    for i, d in enumerate(DATASETS):
        st = [s for s in js["strata"] if s["dataset"] == d]
        for j, s in enumerate(st):
            lab = DS_LABEL[d] if j == 0 else ""
            seeds = ", ".join(str(p["seed"]) for p in s["per_seed"])
            if len(s["per_seed"]) != s["n_seeds"]:
                raise ValueError(f"stratum {s['stratum']}: n_seeds mismatch")
            mean = sum(p["round100_acc_pct"] for p in s["per_seed"]) / s["n_seeds"]
            if abs(mean - s["mean"]) > 1e-9:
                raise ValueError(f"stratum {s['stratum']}: mean mismatch")
            if s["n_seeds"] > 1:
                sd = math.sqrt(sum((p["round100_acc_pct"] - mean) ** 2 for p in s["per_seed"]) / (s["n_seeds"] - 1))
                if abs(sd - s["sd"]) > 1e-9:
                    raise ValueError(f"stratum {s['stratum']}: sd mismatch")
            val = f"{s['mean']:.2f} $\\pm$ {s['sd']:.2f}" if s["sd"] is not None else f"{s['mean']:.2f} (single run)"
            lines.append(f"{lab} & \\texttt{{{s['stratum'][:6]}}} & {seeds} & {s['n_seeds']} & {val} \\\\")
        if i < 2:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tableA_ra_strata.tex", lines)


# ----------------------------------------------------------------------------- Table 5 (new, revision 5)
def table5_fourarm(fa_primary, fa_contrasts):
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data_fourarm/fourarm_primary_summary.csv and",
        "% data_fourarm/fourarm_primary_contrasts.json -- do not edit by hand",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Four-arm controlled experiment (new; separate from the retrospective reanalysis of",
        "Table~\\ref{tab:canonical}--\\ref{tab:paired}): FedAvg and InfoOpt-Full, each run with AIB off and",
        "on, isolating the aggregation rule from the AIB architecture change within this experiment.",
        "Partitions and cohort schedules are identical across all four arms of a dataset-seed by",
        "construction and verified at runtime (no membership assumption). (a) Terminal round-100 test",
        "accuracy (\\%), mean $\\pm$ sample SD, $n=5$ seeds per cell. (b) InfoOpt-Full$-$FedAvg contrasts",
        "within each AIB setting; 95\\% paired-$t$ CIs; Holm--Bonferroni across the family of six.}",
        "\\label{tab:fourarm}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\multicolumn{4}{l}{\\emph{(a) Primary summary (\\%), $n=5$ each}}\\\\",
        "\\textbf{Arm} & \\textbf{CIFAR-100} & \\textbf{EuroSAT} & \\textbf{PathMNIST} \\\\",
        "\\midrule",
    ]
    for a in FOURARM_ARMS:
        cells = []
        for d in DATASETS:
            r = fa_primary[(d, a)]
            cells.append(f"{r['mean']:.2f} $\\pm$ {r['sd']:.2f} ({r['n']})")
        lines.append(f"{FOURARM_ARM_LABEL[a]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{4pt}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\multicolumn{7}{l}{\\emph{(b) InfoOpt-Full$-$FedAvg contrasts within each AIB setting, Holm $m=6$}}\\\\",
        "\\textbf{Dataset} & \\textbf{AIB} & $n$ & $\\Delta$ \\textbf{(pp)} & \\textbf{95\\% CI} & "
        "$p_{\\mathrm{raw}}$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for i, d in enumerate(DATASETS):
        for j, aib in enumerate(["aiboff", "aibon"]):
            c = [x for x in fa_contrasts if x["dataset"] == d and x["aib"] == aib][0]
            lo, hi = c["ci95_low_pp"], c["ci95_high_pp"]
            lab = DS_LABEL[d] if j == 0 else ""
            aib_lab = "off" if aib == "aiboff" else "on"
            lines.append(f"{lab} & {aib_lab} & {c['n']} & {signed(c['mean_difference_pp'])} & "
                         f"[{signed(lo)}, {signed(hi)}] & {c['p_raw']:.3f} & {c['p_holm']:.3f} \\\\")
        if i < 2:
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{3pt}\\begin{minipage}{0.92\\textwidth}\\footnotesize",
        "All six intervals include zero; none of the six primary contrasts is significant after Holm",
        "adjustment. This experiment addresses the architecture confound for FedAvg versus InfoOpt-Full",
        "only; it does not resolve RA-InfoOpt configuration heterogeneity or FedDyn/SCAFFOLD implementation",
        "provenance (Section~\\ref{subsec:provenance}). The aggregation$\\times$AIB interaction is reported",
        "separately in Supplementary Table~A.4.",
        "\\end{minipage}",
        "\\end{table*}",
    ]
    write("table5_fourarm.tex", lines)


# ------------------------------------------------------------------- Table A.4 (new, revision 5, supplement)
def tableA4_fourarm_interaction(fa_interactions):
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data_fourarm/fourarm_aib_interactions.json --",
        "% do not edit by hand",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Four-arm experiment: aggregation $\\times$ AIB interaction, defined as (InfoOpt-Full on",
        "$-$ InfoOpt-Full off) $-$ (FedAvg on $-$ FedAvg off), equivalently (InfoOpt-Full$-$FedAvg on)",
        "$-$ (InfoOpt-Full$-$FedAvg off). Paired $n=5$; a separate prespecified Holm family of three. This",
        "is a prespecified secondary analysis relative to the primary family in main Table~5; it does not",
        "by itself establish that InfoOpt-Full is worse within either AIB setting, since those",
        "primary contrasts are individually uncertain (both intervals include zero).}",
        "\\label{app:tab:fourarm_interaction}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Interaction (pp)} & \\textbf{95\\% CI} & $p_{\\mathrm{raw}}$ & "
        "$p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for d in DATASETS:
        c = [x for x in fa_interactions if x["dataset"] == d][0]
        lo, hi = c["ci95_low_pp"], c["ci95_high_pp"]
        sig = "$^{\\ast}$" if (lo > 0 or hi < 0) else ""
        lines.append(f"{DS_LABEL[d]} & {signed(c['mean_difference_pp'])}{sig} & "
                     f"[{signed(lo)}, {signed(hi)}] & {c['p_raw']:.4f} & {c['p_holm']:.4f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{3pt}\\begin{minipage}{0.85\\textwidth}\\footnotesize",
        "$^{\\ast}$95\\% CI excludes zero after Holm adjustment within this family of three (CIFAR-100 only).",
        "\\end{minipage}",
        "\\end{table}",
    ]
    write("tableA4_fourarm_interaction.tex", lines)


# ----------------------------------------------------------------------------- Table 6 (new, revision 7)
def table6_ra_matched(ra_primary, ra_contrasts):
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data_ra/ra_primary_summary.csv and",
        "% data_ra/ra_primary_contrasts.json -- do not edit by hand",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Matched RA-InfoOpt experiment (new; separate from the retrospective reanalysis of",
        "Table~\\ref{tab:canonical}--\\ref{tab:paired} and from the revision-5 four-arm experiment of",
        "Table~\\ref{tab:fourarm}): 60 fresh runs, three datasets $\\times$ five seeds (100--104)",
        "$\\times$ four arms, all AIB enabled, matched training settings, verified shared partition indices",
        "and cohort schedules, early stopping disabled, terminal round-100 test accuracy. The protocol and",
        "analysis plan were frozen before launch (not an externally registered preregistration).",
        "(a) Terminal accuracy (\\%), mean $\\pm$ sample SD, $n=5$ seeds per cell. (b) Primary family: RA-InfoOpt",
        "with counterfactual probes disabled, minus FedAvg and minus InfoOpt-Full, across three datasets;",
        "95\\% CIs are individual (non-simultaneous) paired-$t$ intervals; Holm--Bonferroni correction is",
        "applied across this family of six. An individual CI excluding zero does not by itself survive the",
        "Holm adjustment; both quantities are reported separately and should not be conflated. This",
        "experiment does not recover or certify the historical five-seed RA-InfoOpt estimate withheld in",
        "Table~\\ref{tab:canonical} (Section~\\ref{subsec:provenance}), and does not resolve the separate",
        "FedDyn/SCAFFOLD implementation-provenance issue. The probe-on/off comparison is reported separately",
        "in Supplementary Table~A.5.}",
        "\\label{tab:ra_matched}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\multicolumn{4}{l}{\\emph{(a) Primary summary (\\%), $n=5$ each}}\\\\",
        "\\textbf{Arm} & \\textbf{CIFAR-100} & \\textbf{EuroSAT} & \\textbf{PathMNIST} \\\\",
        "\\midrule",
    ]
    for a in RA_ARMS:
        cells = []
        for d in DATASETS:
            r = ra_primary[(d, a)]
            cells.append(f"{r['mean']:.2f} $\\pm$ {r['sd']:.2f} ({r['n']})")
        lines.append(f"{RA_ARM_LABEL[a]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{4pt}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\multicolumn{7}{l}{\\emph{(b) RA-InfoOpt (probe off) primary contrasts, individual 95\\% CIs, Holm $m=6$}}\\\\",
        "\\textbf{Dataset} & \\textbf{Contrast} & $n$ & $\\Delta$ \\textbf{(pp)} & \\textbf{95\\% CI} & "
        "$p_{\\mathrm{raw}}$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    sig_notes = []
    for i, d in enumerate(DATASETS):
        for j, base in enumerate(["ra_probeoff minus fedavg_aibon", "ra_probeoff minus infoopt_aibon"]):
            c = [x for x in ra_contrasts if x["dataset"] == d and x["contrast"] == base][0]
            lo, hi = c["ci95_pp"]
            lab = DS_LABEL[d] if j == 0 else ""
            name = "RA-InfoOpt(off)$-$FedAvg" if "fedavg" in base else "RA-InfoOpt(off)$-$InfoOpt-Full"
            sig = "$^{\\ast}$" if (lo > 0 or hi < 0) else ""
            lines.append(f"{lab} & {name} & {c['n']} & {signed(c['mean_diff_pp'])}{sig} & "
                         f"[{signed(lo)}, {signed(hi)}] & {c['raw_p']:.4f} & {c['holm_p']:.4f} \\\\")
            if lo > 0 or hi < 0:
                sig_notes.append(f"{DS_LABEL[d]} {name.replace('$-$', '$-$')}")
        if i < 2:
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{3pt}\\begin{minipage}{0.92\\textwidth}\\footnotesize",
        "$^{\\ast}$Individual unadjusted 95\\% CI excludes zero (CIFAR-100 RA$-$FedAvg: "
        "+0.208~pp, $p_{\\mathrm{Holm}}=0.189$; EuroSAT RA$-$InfoOpt-Full: +6.681~pp, "
        "$p_{\\mathrm{Holm}}=0.206$). Neither survives Holm correction across the family of six, so neither",
        "is a significant result at the family level; the smallest Holm-adjusted $p$-value in this family is",
        "approximately 0.189. These two flagged intervals should not be read as RA-InfoOpt consistently",
        "outperforming either control, and the remaining, zero-crossing intervals should not be read as",
        "demonstrating equivalence.",
        "\\end{minipage}",
        "\\end{table*}",
    ]
    write("table6_ra_matched.tex", lines)


# ------------------------------------------------------------------- Table A.5 (new, revision 7, supplement)
def tableA5_ra_probe(ra_probe_contrasts):
    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py from data_ra/ra_probe_contrasts.json -- do not edit",
        "% by hand",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Matched RA-InfoOpt experiment: probe-on minus probe-off, one contrast per dataset, a",
        "separate prespecified Holm family of three (distinct from the primary family of main Table~6).",
        "This comparison measures instrumentation sensitivity of enabling the",
        "counterfactual probe evaluation, including the probe model's initialization effect on the random",
        "number generator stream, which is not isolated from other probe-related effects in the existing",
        "implementation. It is not a comparison between two distinct aggregation mechanisms and does not by",
        "itself isolate a pure probe-evaluation causal effect. The probe subset overlaps the training data",
        "and was not used to select configurations or endpoints. All three individual 95\\% CIs include",
        "zero, and none of the three comparisons survives Holm correction.}",
        "\\label{app:tab:ra_probe}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Probe on$-$off (pp)} & \\textbf{95\\% CI} & $p_{\\mathrm{raw}}$ & "
        "$p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for d in DATASETS:
        c = [x for x in ra_probe_contrasts if x["dataset"] == d][0]
        lo, hi = c["ci95_pp"]
        lines.append(f"{DS_LABEL[d]} & {signed(c['mean_diff_pp'])} & "
                     f"[{signed(lo)}, {signed(hi)}] & {c['raw_p']:.4f} & {c['holm_p']:.4f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{3pt}\\begin{minipage}{0.85\\textwidth}\\footnotesize",
        "Protocol and verification detail: 60 runs, all four arms AIB-enabled, matched training settings",
        "(learning rate 0.01, momentum 0.9, weight decay 0.0005, two local epochs, batch size 64, gradient",
        "clip 5, ten clients with five selected per round, Dirichlet $\\alpha=0.3$, minimum 50 samples per",
        "client), shared partition indices and cohort schedules generated once per dataset/seed and verified",
        "identical across all four arms of that cell, early stopping disabled, 100 rounds, no failed",
        "attempts recorded. 537 collection-file hashes were checked server-side; checkpoint strict-loading",
        "and tensor-finiteness checks rely on the collected server verification reports, since model",
        "checkpoints were not themselves supplied to this analysis workspace. The raw execution manifest",
        "carries a stale metadata label, \\texttt{four\\_arm\\_20261009}, left over from an earlier naming",
        "scheme; it is preserved unaltered in the frozen manifest and documented here as a metadata issue,",
        "not used for run selection, and should not be mistaken for the revision-5 four-arm experiment of",
        "main Table~5, which is a distinct, separately frozen dataset.",
        "\\end{minipage}",
        "\\end{table}",
    ]
    write("tableA5_ra_probe.tex", lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    verify_fourarm_package()
    primary = load_primary()
    _label, sens_summ, sens_contr = load_sensitivity()
    cs = load_contrasts()
    table3(primary, load_excluded_cell_sensitivity())
    table4(cs)
    tableA_primary(cs)
    tableA_sensitivity(cs, sens_summ, sens_contr)
    tableA_strata()
    fa_primary = load_fourarm_primary()
    fa_contrasts = load_fourarm_contrasts()
    fa_interactions = load_fourarm_interactions()
    table5_fourarm(fa_primary, fa_contrasts)
    tableA4_fourarm_interaction(fa_interactions)
    verify_ra_package()
    ra_primary = load_ra_primary()
    ra_contrasts = load_ra_contrasts()
    ra_probe_contrasts = load_ra_probe_contrasts()
    table6_ra_matched(ra_primary, ra_contrasts)
    tableA5_ra_probe(ra_probe_contrasts)
    verify_stepsize_package()
    stepsize_decision = load_stepsize_decision()
    load_stepsize_exploratory()
    load_stepsize_buffer()
    tableA6_stepsize_pilot(stepsize_decision)
