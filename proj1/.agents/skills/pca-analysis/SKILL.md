# PCA Analysis Skill

## Purpose

Use this skill when PCA is appropriate for summarizing variation in a dataset with a linear, lower-dimensional representation.

PCA is useful as a global linear baseline and when understanding how much variation can be represented by a smaller number of components is important.

## When to Consider PCA

Consider PCA when:

- the main goal is to summarize global variation using a linear representation;
- correlated features may be represented by fewer linear combinations;
- explained variance is useful for assessing how much information is retained.

PCA may be less appropriate when the important structure is strongly nonlinear.

Compared with nonlinear methods, prefer PCA when a linear representation is appropriate or when a simple global baseline is useful.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Run:

   ```bash
   python3 analysis/run_pca.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy>
   ```

3. Inspect:

   - `pca_metrics.json`
   - `pca_train.csv`
   - `pca_eval.csv`
   - `pca_train_2d.png`
   - `pca_eval_2d.png`

4. Use the PCA results together with embedding evaluation and visualization to determine whether PCA provides a useful representation.

## PCA Output

The implementation fits PCA using the training data and applies the fitted transformation to evaluation data.

All available principal components are computed internally so that explained variance can be assessed.

The saved embeddings contain only the first two principal components for visualization and comparison with other dimension-reduction methods.

Pay attention to:

- variance explained by the first two components;
- cumulative explained variance;
- number of components required to explain 90% of the variance;
- structure visible in the two-dimensional embedding.

A low two-dimensional explained variance does not by itself mean that PCA failed. It indicates that the first two components represent only a limited portion of the total variation.

## Interpretation

Interpret PCA as a linear representation of major directions of variation.

Do not interpret separation in the PCA plot as evidence of distinct populations without additional support.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting PCA.

Compare PCA with nonlinear methods when the inspection results or analysis objective suggest that important structure may not be linear.

## Reporting

Report why PCA was selected, the preprocessing used, and the most relevant explained-variance results.

Describe important visible structure and limitations without overstating what the embedding demonstrates.