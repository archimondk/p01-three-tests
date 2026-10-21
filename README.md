# P01 three-test screen: frozen analysis pipeline (v2.0)

A single pipeline applying three tests — **analytic availability**, **reliability**, and
**statistical comparability** — to healthcare-associated infection (HAI) surveillance
indicators, and the decision rules used to judge whether an indicator may be used for
inter-hospital comparison.

This repository holds the code and the protocol. It contains no data: the input files are
institutional aggregate surveillance returns held by the provincial quality-control centre
and are not publicly available.

## Contents

| Path | What it is |
|---|---|
| `p01_three_tests/` | The analysis package. `p01_three_tests/PROTOCOL.md` is the authoritative protocol. |
| `p01_three_tests/PROTOCOL.md` | Protocol: the decisions D1–D7, the data and cleaning rules, the three tests, the combination rules, the consequence analysis, and the record of deviations. |
| `FREEZE_CHECK_v2.0.md` | Pre-freeze verification: test results, comparison against the previous manuscript version, D1–D7 evidence, the post-receipt comparisons, and the data delivery log with input hashes. |
| `v16_code_review.md`, `p01_three_tests/docs/v16_code_review.md` | Code review of the earlier, pre-freeze script set. |
| `PROTOCOL.md` | A copy of the protocol, kept in sync with `p01_three_tests/PROTOCOL.md`. |

## Freeze

The decision rules, thresholds, cleaning rules, bootstrap design and seeding were fixed in
tag **`p01-three-tests-freeze-v2.0`** on 19 October 2026, 22:26 +0800, before the updated
quarterly extract was received (2026Q2 re-export at 22:37 and 2026Q3 at 22:44, the same
day).

These rules were **not preregistered**. They were derived from a data audit of an earlier
extract (2025Q1–2026Q2) and then frozen before the updated extract was analysed. Because
the two extracts overlap in 2025Q1–2026Q1, the updated results are a consistency check
rather than an independent validation.

Verification that the freeze held: the pre-freeze run (earlier extract) and the final
formal run (updated extract) gave identical results for every indicator — analytic
availability, ICC with bootstrap intervals, the three test verdicts, the overall verdicts,
event counts, overdispersion factors and precision thresholds. Details in
`FREEZE_CHECK_v2.0.md`.

## The pre-freeze package

Tag `p01-three-tests-freeze-v2.0` contains `p01_three_tests_freeze_v2.0.zip`, the original
package as delivered to the study team before the freeze, kept unmodified as a record of
what was handed over. It is **superseded and must not be used to reproduce the reported
results**: it predates the C01b cleaning rule and the two code fixes described in
`FREEZE_CHECK_v2.0.md` §2, and it reports nothing about the data that arrived afterwards.

The zip is not part of the current tree. The working tree is the version to use.

## Requirements

- Python ≥ 3.10
- pandas, numpy, scipy, matplotlib, openpyxl, statsmodels, pytest

## Running

```bash
cd p01_three_tests

# 1. Tests (45; the demonstration-model and MixedLM cross-check cases are
#    skipped when statsmodels is unavailable)
python -m pytest tests

# 2. Trial run — results must not enter the manuscript
python run.py --raw <data directory> --B 200

# 3. Formal run
python run.py --raw <data directory> \
              [--code-register code_register.xlsx] \
              [--custodian-flags custodian_flags.csv]

# 4. Compare two extracts
python revision_audit.py --old <old data directory> --new <new data directory>
```

The data directory holds the platform's quarterly returns, one file per quarter, named
`t_YY_S_Q.csv` (`t_25_S_1.csv` = 2025 Q1). Every file must match the pattern
`^t_(\d{2})_S_([1-4])\.csv$`; files that do not are silently ignored, so check that each
quarter is present exactly once before running. A file that fails the required-field
check stops the run rather than being merged with missing columns.

`--code-register` points to the institutional code register (column 2 holds the hospital
code) used to count institutions activated on the platform. `--custodian-flags` takes a
CSV or Excel file with columns `ID, period, indicator, flag` (`flag` is `good` or `bad`;
`indicator` is an indicator key or `composite`).

## Outputs

Each run writes to `outputs/run_<timestamp>/`:

| Output | Contents |
|---|---|
| `tables/*.csv` and `results.xlsx` | Tables 1–5 and S1–S12, plus process tables (X_*) |
| `figures/` | Figures 1–4 and S1–S4, as PNG and TIFF, 300 dpi |
| `numbers.json` | Every number quoted in the manuscript; the only permitted source |
| `manifest.json` | Input file hashes, code hashes, git tag, software versions, bootstrap count and seed |

Figures are written to `figures/` only when `--skip-figures` is not given. Run outputs are
excluded from version control; `manifest.json` records the input and code hashes needed to
reproduce a run.

## Files in the package

| File | Role |
|---|---|
| `config.py` | Frozen parameters. After the freeze only `PATHS`, `CUSTODIAN` and `COLUMN_ALIASES` may be changed; any other change is a protocol deviation and requires a new tag. |
| `data.py` | Reading, validation, cleaning, reporting-status coding |
| `core.py` | REML ICC, cluster bootstrap, Spearman–Brown, verdict rules, overdispersion, DerSimonian–Laird estimator (no statsmodels required) |
| `analysis.py` | The three tests, Tables 1–3 and S1–S9, sensitivity analyses |
| `models.py` | Demonstration models and the MixedLM cross-check (requires statsmodels) |
| `consequence.py` | Consequence analysis: rank uncertainty intervals, agreement between crude ranking and adjusted methods, naive ranking, comparison with a quality-control list |
| `figures.py` | Figures |
| `run.py` | Entry point |
| `revision_audit.py` | Comparison of two extracts |
| `tests/` | Synthetic data generator and 45 tests |
| `docs/v16_code_review.md` | Code review of the pre-freeze script set |

## Citation

If you use this code, please cite the protocol and the verification record, and state the
git tag you used. All manuscript numbers must come from a single formal run: take them
from that run's `numbers.json` and `tables/`, never from more than one run or from
different scripts.
