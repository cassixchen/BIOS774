# Data Inspection

## Method

Read `DATA.md` and `schema.json` when available. Run:

```bash
.venv/bin/python3 analysis/inspect_data.py \
  --data-dir <DATA_DIR> \
  --output-dir <OUTPUT_DIR>
```

Inspect `data_profile.json` for sample size, feature types, missing values, sparsity, scales, identifiers, labels, and row totals.

## Interpretation

Use observed characteristics and feature meaning to guide preprocessing and model selection. Exclude identifiers and avoid treating labels as unsupervised input features.

Report only findings relevant to modeling decisions.
