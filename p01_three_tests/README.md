# P01 three-test screen: frozen analysis pipeline

One pipeline that merges the earlier script set (`audit.py`, `p1`, `p5`, `p7`–`p10`) into a
single flow. All parameters live in `config.py`; the rules are in `PROTOCOL.md`.

The pipeline applies three tests — analytic availability, reliability and statistical
comparability — to HAI surveillance indicators, and decides whether each indicator may be
used for single-quarter inter-hospital comparison.

## Requirements

- Python ≥ 3.10
- pandas, numpy, scipy, matplotlib, openpyxl, statsmodels, pytest

## Running

```bash
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

Run outputs are excluded from version control; `manifest.json` records the input and code
hashes needed to reproduce a run.

## Files

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
