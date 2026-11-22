# InfoOpt Public Release — Stage 2C Decision

## Decision

**GO** for private GitHub upload, with three small pre-upload conditions.

## Reasoning

All critical CPU-only reproducibility checks pass:

- Table generation and all 39 table-regression tests pass.
- Independent numerical recomputation of all reported contrasts and
  summaries passes.
- FedDisco extension tests pass (28/28), including independent weight
  reconstruction from saved diagnostics.
- SHA-256 manifest is regenerated and self-verifies.
- No syntax errors in any Python file.
- No credentials or private keys are present in the release.

The only blocked checks are those that require the frozen LaTeX
manuscript source, which was not present in the workspace. These checks
are non-critical for the numerical reproducibility of the reported
results; they verify prose wording, cross-references, and the absence of
superseded values in the manuscript text.

## Conditions before private GitHub upload

1. **License.** Replace the placeholder `LICENSE` file with the final
   author-approved open-source license text (recommended: Apache-2.0 or
   MIT).
2. **Manuscript source (optional but recommended).** Add the frozen
   LaTeX source files to `manuscript/` and confirm that
   `check_numbers.py` passes all checks, including prose and reference
   checks.
3. **Historical-path notice.** Add a brief note to the top-level README
   or `RELEASE_WARNINGS.txt` stating that some historical audit-only
   files contain machine-specific paths and should not be executed
   without review.

## Not ready for

- **Public release or DOI** until the license placeholder is replaced.
- **Claiming 100% manuscript-verification coverage** until the LaTeX
  source is added and `check_numbers.py` runs without skipped checks.

## Next step

Obtain author/institutional license approval, replace `LICENSE`, and
optionally add the manuscript source. After those steps the repository
is ready for a private GitHub repository upload.
