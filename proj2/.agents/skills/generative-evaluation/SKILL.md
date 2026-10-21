
# Generative Evaluation

## Method

Use `analysis/evaluate_generation.py` for each trained model:

```bash
.venv/bin/python3 analysis/evaluate_generation.py \
  --data-dir <DATA_DIR> \
  --model-dir <MODEL_DIR> \
  --output-dir <OUTPUT_DIR>
```

Dataset modality is detected automatically. Inspect `evaluation_metrics.json` and generated samples.

## Interpretation

- Images: compare color statistics, spatial variation, diversity, and nearest-neighbor distances; inspect image grids.
- scRNA-seq: compare library sizes, sparsity, detected genes, expression differences, and sample distances.

Interpret metrics relative to real-data baselines and the user objective. Identify possible mode collapse, memorization, unrealistic samples, or distorted count distributions.

Use held-out data only for evaluation. Nearest-neighbor distances are not privacy guarantees; summary statistics alone do not establish biological realism or downstream utility.

If the objective requires evaluation that existing tools cannot perform, explicitly report the limitation rather than claiming success.

## Recommendation

Recommend `RETAIN_MODEL`, `IMPROVE_MODEL`, `TRY_ALTERNATIVE`, or `DO_NOT_USE_SYNTHETIC_DATA`, with reasons grounded in the objective and observed results.

Explain appropriate downstream uses and any restrictions.
