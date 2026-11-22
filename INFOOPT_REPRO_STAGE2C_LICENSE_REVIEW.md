# InfoOpt Public Release — Stage 2C License Review

## Scope

Review of authorship, existing license notices, third-party code, and
recommendation for the public reproducibility package located at
`D:\InfoOpt Latex\Code\InfoOpt_Public_Release`.

## Findings

### Original code

- The manuscript authors wrote the federated-learning training code,
  analysis scripts, figure-generation scripts, and FedDisco extension
  implementation.
- A grep for `Copyright`, `License`, `SPDX`, `MIT`, `Apache`, `GPL`,
  `BSD`, `All rights reserved`, and `Redistribution` across
  `source_snapshots/` and `feddisco_extension/snapshot/` returned **no
  matches**.
- No explicit open-source license has been applied to the authors' code.

### Third-party content

- The repository uses standard open-source scientific-Python dependencies
  (PyTorch, torchvision, NumPy, SciPy, PyYAML, pandas, matplotlib) listed
  in `requirements.txt`. These are runtime dependencies, not bundled code.
- `elsarticle-num.bst` is an official Elsevier bibliography style file
  supplied by the authors; it is a standard publisher-supplied file.
- No copied third-party code blocks, vendored libraries, or modified
  upstream files were identified in the release source.

### Patent, institutional, or publisher restrictions

- No patent markings or institutional ownership statements were found in
  the source files.
- The manuscript has not been published at the time of this review, so
  publisher copyright-transfer terms do not yet apply. The authors retain
  copyright until a publication agreement is signed.

## Recommendation

Because the code is original work and no conflicting license terms were
found, an OSI-approved permissive license is appropriate. The two
standard choices are:

1. **MIT License** — simple, permissive, widely recognized in machine
   learning repositories.
2. **Apache-2.0** — permissive but includes an explicit patent grant and
   is the default license for many PyTorch-ecosystem projects.

**Recommended:** Apache-2.0, to align with common practice in the
federated-learning and PyTorch communities and to provide a clear patent
grant.

## Current action

- A placeholder `LICENSE` file has been added to the release root stating
  that the license is "to be determined before public release."
- **The placeholder must be replaced with the final license text before
  any public release or GitHub upload.**
- Final license selection requires confirmation from all authors and/or
  the relevant institution.

## Risk

- **Low:** no third-party copyleft code was found that would constrain
  the license choice.
- **Medium (procedural):** public release without a license creates
  legal ambiguity. Do not upload to a public repository until the
  placeholder is replaced.
