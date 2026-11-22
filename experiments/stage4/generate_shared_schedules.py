#!/usr/bin/env python3
"""Stage-4 shared inputs: copy the FROZEN Stage-1 alpha=0.3 partitions
and cohort schedules into the Stage-4 namespace, with hash verification
against the Stage-1 SHARED_MANIFEST and a recorded provenance trail.

Stage 4 reuses the original Stage-1 shared inputs EXACTLY. This script
never generates replacements: if a source file or its recorded hash is
missing, it aborts and prints the exact files to request.

Source: --stage1-shared-dir (default: the package's
stage1_shared_reference/, extracted hash-verified from the Stage-1
writing package). Destination: --shared-dir (EXP_ROOT/shared at launch),
filenames partition_<ds>_alpha0.3_seed<s>.json and
cohort_schedule_<ds>_alpha0.3_seed<s>.json (byte-identical content,
write-once). Each copied partition is re-validated (coverage,
duplicates, range, client count, minimum size).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (100, 101, 102, 103, 104)
DATASETS = ("cifar100", "eurosat", "pathmnist")
ALPHA = "0.3"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def validate_partition(indices, num_clients, min_samples):
    """Hard checks on a copied partition. Raises ValueError."""
    if set(indices.keys()) != {str(i) for i in range(num_clients)}:
        raise ValueError(f"client keys != '0'..'{num_clients - 1}'")
    flat = [i for v in indices.values() for i in v]
    if len(flat) != len(set(flat)):
        raise ValueError("duplicate indices in partition")
    if sorted(flat) != list(range(len(flat))):
        raise ValueError("partition does not cover exactly 0..N-1")
    sizes = {k: len(v) for k, v in indices.items()}
    if min(sizes.values()) < min_samples:
        raise ValueError(f"client below min_samples: {sizes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-shared-dir",
                    default=str(HERE / "stage1_shared_reference"))
    ap.add_argument("--shared-dir", default=str(HERE / "shared"))
    ap.add_argument("--configs-dir", default=None,
                    help="accepted for launcher compatibility; the "
                         "backbone plays no role in shared inputs")
    cli = ap.parse_args()
    src_root = Path(cli.stage1_shared_dir)
    dst_root = Path(cli.shared_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    src_manifest_path = src_root / "SHARED_MANIFEST.json"
    if not src_manifest_path.is_file():
        raise SystemExit(
            f"Stage-1 SHARED_MANIFEST.json missing in {src_root}. "
            f"Request the original Stage-1 shared files "
            f"(evidence/shared/*).")
    src_manifest = json.loads(src_manifest_path.read_text(
        encoding="utf-8"))
    recorded = {}
    for e in src_manifest["files"]:
        for kind in ("partition", "schedule"):
            recorded[Path(e[kind]["path"]).name] = e[kind]["sha256"]

    subset = set(__import__("os").environ.get("DATASETS", "").split())
    datasets = [d for d in DATASETS if not subset or d in subset]

    out_entries = []
    missing = []
    for ds in datasets:
        for seed in SEEDS:
            for kind, src_name, dst_name in (
                    ("partition", f"partition_{ds}_seed{seed}.json",
                     f"partition_{ds}_alpha{ALPHA}_seed{seed}.json"),
                    ("schedule",
                     f"cohort_schedule_{ds}_seed{seed}.json",
                     f"cohort_schedule_{ds}_alpha{ALPHA}_seed{seed}.json")):
                src = src_root / src_name
                if not src.is_file():
                    missing.append(str(src))
                    continue
                blob = src.read_bytes()
                src_hash = sha256_bytes(blob)
                want = recorded.get(src_name)
                if want is None or want != src_hash:
                    raise SystemExit(
                        f"source hash mismatch for {src_name}: recorded "
                        f"{want}, actual {src_hash}. The Stage-1 shared "
                        f"input cannot be trusted; request the original "
                        f"file.")
                dst = dst_root / dst_name
                if dst.exists():
                    if dst.read_bytes() != blob:
                        raise SystemExit(f"existing file differs, refusing "
                                         f"to overwrite: {dst}")
                else:
                    dst.write_bytes(blob)
                entry = {"dataset": ds, "alpha": ALPHA, "seed": seed,
                         kind: {"path": str(dst), "sha256": src_hash,
                                "source": src_name,
                                "source_sha256": src_hash}}
                out_entries.append((ds, seed, kind, entry, blob))
    if missing:
        raise SystemExit("MISSING Stage-1 shared files — request these "
                         "exact files, do not regenerate:\n"
                         + "\n".join(missing))

    # re-validate copied partitions (coverage/dupes/range/count/min size)
    import numpy as np  # noqa: F401  (kept for parity with stage3 checks)
    for ds, seed, kind, entry, blob in out_entries:
        if kind != "partition":
            continue
        payload = json.loads(blob)
        validate_partition(payload["client_indices"],
                           payload["num_clients"], 50)

    # merge entries per (ds, seed)
    merged = {}
    for ds, seed, kind, entry, blob in out_entries:
        rec = merged.setdefault((ds, seed),
                                {"dataset": ds, "alpha": ALPHA,
                                 "seed": seed})
        rec[kind] = entry[kind]
    provenance = {
        "generated_by": "generate_shared_schedules.py (stage4: "
                        "verified copy, NOT regeneration)",
        "experiment": "stage4_backbone_20261014",
        "alpha": ALPHA,
        "source_stage1_shared_dir": str(src_root),
        "source_stage1_manifest_sha256": sha256_bytes(
            src_manifest_path.read_bytes()),
        "note": ("Byte-identical copies of the frozen Stage-1 alpha=0.3 "
                 "partitions and cohort schedules, hash-verified against "
                 "the Stage-1 SHARED_MANIFEST. Partitions re-validated "
                 "for coverage/duplicates/range/count/min-size."),
        "seeds": list(SEEDS),
        "files": [merged[k] for k in sorted(merged)],
    }
    mpath = dst_root / "SHARED_MANIFEST.json"
    blob = json.dumps(provenance, indent=1, sort_keys=True).encode()
    if mpath.exists():
        if mpath.read_bytes() != blob:
            raise SystemExit(f"existing shared manifest differs, "
                             f"refusing to overwrite: {mpath}")
    else:
        mpath.write_bytes(blob)
    print(f"copied {len(out_entries)} Stage-1 shared files into "
          f"{dst_root} (provenance recorded)")


if __name__ == "__main__":
    main()
