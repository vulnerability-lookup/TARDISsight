# Borrowing Strength Across Vulnerabilities (VulnOptiCON 2026 draft)

Follow-up to *Modeling Sparse and Bursty Vulnerability Sightings* (arXiv:2604.16038).
Working title — change `\title` in `borrowing-strength-sightings.tex` and rename
this folder if desired.

## Build

```bash
pdflatex borrowing-strength-sightings.tex
bibtex   borrowing-strength-sightings
pdflatex borrowing-strength-sightings.tex
pdflatex borrowing-strength-sightings.tex
```

Produces `borrowing-strength-sightings.pdf`.

## Contents

- `borrowing-strength-sightings.tex` — the paper.
- `mybib.bib` — references.
- `figures/` — figures, copied from `docs/img/eval/` in the repo root and
  regenerable with `python -m tardissight.plots`.

The tables and figures correspond to the results in `docs/evaluation.md`,
`docs/pooling.md`, `docs/typed.md`, `docs/bayesian.md`, and `docs/zeroinflated.md`.
