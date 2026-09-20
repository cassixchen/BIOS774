# UMAP Analysis Skill

## Purpose

Use this skill when UMAP is appropriate for producing a nonlinear representation that emphasizes local neighborhood structure.

UMAP constructs a neighborhood-based representation that can be useful for exploratory visualization of high-dimensional data.

## When to Consider UMAP

Consider UMAP when:

- nonlinear local neighborhood structure is important;
- exploratory visualization is a major objective;
- a flexible neighborhood-based representation is useful;
- out-of-sample transformation for evaluation data is valuable.

UMAP may be less appropriate when preserving global distances or directly quantifying explained variance is the primary objective.

Compared with more assumption-specific manifold methods, UMAP is useful when local neighborhood preservation is the goal without requiring a locally linear or geodesic-distance interpretation.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Select appropriate UMAP hyperparameters.

3. Run:

   ```bash
   python3 analysis/run_umap.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy> \
     --n-neighbors <n_neighbors> \
     --min-dist <min_dist> \
     --metric <metric> \
     --random-state <random_state>
   ```

4. Inspect:

   - `umap_metrics.json`
   - `umap_train.csv`
   - `umap_eval.csv`
   - `umap_train_2d.png`
   - `umap_eval_2d.png`

5. Use the UMAP results together with embedding evaluation and visualization to determine whether the representation is useful.

## Hyperparameters

`n_neighbors` controls the scale of the neighborhood structure emphasized by UMAP. Smaller values emphasize more local structure, while larger values incorporate broader neighborhoods.

`min_dist` controls how closely observations can be placed in the embedding. Smaller values allow tighter groupings, while larger values encourage a more spread-out representation.

`metric` determines how distances between observations are calculated.

Use `random_state` to make the analysis reproducible.

Choose hyperparameters based on the data characteristics and analysis objective rather than assuming the defaults are optimal.

If the result appears sensitive to important hyperparameters, consider a limited sensitivity analysis using another reasonable configuration.

## UMAP Output

The implementation fits UMAP using the training data and applies the fitted transformation to evaluation data.

Pay attention to:

- train and evaluation trustworthiness;
- structure visible in the two-dimensional embeddings;
- sensitivity to important hyperparameters when relevant.

## Interpretation

Interpret UMAP primarily as a nonlinear representation of neighborhood structure.

Do not assume that distances between separated groups in the embedding accurately represent global distances in the original data.

Visible groups should not automatically be interpreted as distinct populations without additional evidence.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting UMAP.

## Reporting

Report why UMAP was selected, the preprocessing and hyperparameters used, and the most relevant evaluation results.

Mention important sensitivity or limitations when they affect interpretation.