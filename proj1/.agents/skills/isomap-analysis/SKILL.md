# Isomap Analysis Skill

## Purpose

Use this skill when Isomap is appropriate for representing nonlinear structure that may lie on a lower-dimensional manifold.

Isomap preserves approximate geodesic distances by constructing a neighborhood graph and measuring distances along that graph.

## When to Consider Isomap

Consider Isomap when:

- nonlinear manifold structure is plausible;
- distances along the underlying manifold are meaningful;
- preserving approximate geodesic distances is relevant to the analysis objective.

Isomap may be less appropriate when the neighborhood graph does not provide a reliable approximation of manifold distances or computational cost is too high.

Compared with other neighborhood methods, prefer Isomap when preserving global manifold geometry through geodesic distances is important.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Select an appropriate neighborhood size and distance metric.

3. Run:

   ```bash
   python3 analysis/run_isomap.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy> \
     --n-neighbors <n_neighbors> \
     --metric <metric>
   ```

4. Inspect:

   - `isomap_metrics.json`
   - `isomap_train.csv`
   - `isomap_eval.csv`
   - `isomap_train_2d.png`
   - `isomap_eval_2d.png`

5. Use the Isomap results together with embedding evaluation and visualization to determine whether the representation is useful.

## Hyperparameters

`n_neighbors` controls the neighborhood graph used to approximate the manifold.

Smaller values emphasize more local relationships but may produce a poorly connected or unstable graph. Larger values create a more connected graph but may make the representation less sensitive to local manifold structure.

`metric` determines how distances between observations are calculated when constructing neighborhoods.

Choose hyperparameters based on the data characteristics and analysis objective rather than assuming the defaults are optimal.

If the result appears sensitive to the neighborhood size, consider a limited sensitivity analysis using another reasonable value.

## Isomap Output

The implementation fits Isomap using the training data and applies the fitted transformation to evaluation data.

Pay attention to:

- train and evaluation trustworthiness;
- reconstruction error;
- structure visible in the two-dimensional embeddings;
- sensitivity to the neighborhood size when relevant.

## Interpretation

Interpret Isomap as a nonlinear representation based on approximate geodesic distances.

Do not assume that visible groups in the embedding represent distinct populations without additional evidence.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting Isomap.

## Reporting

Report why Isomap was selected, the preprocessing and hyperparameters used, and the most relevant evaluation results.

Mention important sensitivity or limitations when they affect interpretation.