#!/usr/bin/env python3
"""Package the Stage-1 preparation directory into a ZIP with guaranteed
POSIX ("/") member names — PowerShell Compress-Archive on some Windows
builds emits backslash members, which extract as literal filenames on
Linux. This packager never does that, and asserts it.

Usage:  python package_zip.py <source_dir> <output_zip>
Refuses to overwrite an existing ZIP.
"""
import sys
import zipfile
from pathlib import Path


def main():
    src = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite: {out}")
    files = sorted(f for f in src.rglob("*") if f.is_file()
                   and "__pycache__" not in f.parts
                   and not f.name.endswith(".pyc"))
    if not files:
        raise SystemExit("no files to package")
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = f.relative_to(src).as_posix()  # always "/" separators
            assert "\\" not in arc
            z.write(f, arc)
    # verify the written archive
    with zipfile.ZipFile(out) as z:
        bad = [n for n in z.namelist() if "\\" in n]
    if bad:
        out.unlink()
        raise SystemExit(f"archive contains backslash members: {bad[:5]}")
    print(f"wrote {out} ({len(files)} files, POSIX member names)")


if __name__ == "__main__":
    main()
