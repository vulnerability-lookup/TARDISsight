# Borrowing Strength Across Vulnerabilities (VulnOptiCON draft)

Follow-up to *Modeling Sparse and Bursty Vulnerability Sightings* (arXiv:2604.16038).
Working title — change `\title` in `main.tex` and rename this folder if desired.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Produces `main.pdf`.

## Contents

- `main.tex` — the paper.
- `mybib.bib` — references.
- `figures/` — figures, copied from `docs/img/eval/` in the repo root and
  regenerable with `python -m tardissight.plots`.

The tables and figures correspond to the results in `docs/evaluation.md`,
`docs/pooling.md`, and `docs/typed.md`.
