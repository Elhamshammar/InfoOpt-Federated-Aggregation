# Manuscript source files

This directory is reserved for the frozen LaTeX manuscript source used by `analysis/scripts/check_numbers.py`.

The following files are required for a complete numerical cross-check:

- `main_revised.tex` — revised main manuscript (Elsevier `cas-dc` class)
- `supplementary.tex` — revised supplementary material
- `references.bib` — bibliography
- `elsarticle-num.bst` — bibliography style
- `original/main_elsevier_ORIGINAL.tex` — original submitted main manuscript
- `original/Supplementary_Material_ORIGINAL.tex` — original submitted supplement

## How to run the check

From the `analysis/` directory:

```bash
python scripts/check_numbers.py /path/to/build /path/to/manuscript
```

or set the environment variable:

```bash
export INFOOPT_MANUSCRIPT_DIR=/path/to/manuscript
python scripts/check_numbers.py /path/to/build
```

If the manuscript source is not present, `check_numbers.py` will still run all checksum, recomputation, and generated-table checks, and will skip only the prose-quote, superseded-value, cross-reference, and LaTeX-log checks.

## Note

The manuscript source has not been included in this release package because the final Astra-corrected LaTeX files were not found in the workspace during packaging. Add them here before running the full numerical-and-prose verification.
