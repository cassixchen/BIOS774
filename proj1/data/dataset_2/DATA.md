# Dataset 2: PBMC3K

This dataset is derived from the PBMC3K single-cell gene-expression count
matrix.

Each observation represents a cell, and each feature represents the raw
expression count for a selected gene.

The original dataset contains 2,700 cells and 32,738 genes.

Training observations: 2160

Evaluation observations: 540

The cells were split into training and evaluation sets using an 80/20 split
with random seed 42.

Gene filtering and selection were performed using the training observations
only. Genes detected in at least three training cells were retained as
candidates. From these genes, the 2,000 genes with the highest raw-count
variance in the training data were selected as analysis features.

The same 2,000 selected genes were then retained for the evaluation
observations.

The `sample_id` column contains the original cell barcode and should not be
used as an analysis feature.

The dataset does not contain supplied cell-type labels.

The gene-expression features remain raw nonnegative counts. No library-size
normalization, log transformation, standardization, or other analysis
preprocessing was applied when constructing this dataset.

The dataset is sparse, with many zero gene-expression values. The autonomous
analysis should inspect the supplied representation before determining
appropriate preprocessing.
