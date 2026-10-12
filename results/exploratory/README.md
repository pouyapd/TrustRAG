# Exploratory analyses — not part of the submitted manuscript

Everything under `results/exploratory/` was produced after the manuscript was submitted
(23 September 2026). None of it is reported in the manuscript, and none of it replaces a
number in `results/localisation/`, `results/goldspan_adjudication/`, `results/figures/` or
`results/*.json`, which stay as they were at tag `v1.1.0`.

Each subdirectory states its question, how its numbers were produced, how to recompute
them, and what they do and do not show.

| Directory | Question |
|---|---|
| [`nq_reference_sensitivity/`](nq_reference_sensitivity/README.md) | How far do the NQ localisation results depend on scoring against the first annotator's long answer only? |
| [`qasper_reference_sensitivity/`](qasper_reference_sensitivity/README.md) | How far do the QASPER localisation results depend on merging every annotator's evidence? |
| [`reference_sensitivity_claims.md`](reference_sensitivity_claims.md) | Which claims of `results/localisation/README.md` hold when both corpora are scored against the same kind of reference? (`scripts/reference_sensitivity_claims.py`) |
