# Dataset 2: PBMC3K

This dataset is a subset of the PBMC3K single-cell gene-expression count matrix containing cells with available Seurat cell-type annotations.

Each observation is a cell represented by 2,000 selected gene-expression count features.

Training observations: 2110

Evaluation observations: 528

The original PBMC3K dataset contains 2,700 cells and 32,738 genes. Cell-type annotations from the processed Seurat PBMC3K dataset were matched to the original cells using their cell barcodes. Of the 2,700 original cells, 2,638 had corresponding cell-type annotations and were retained for analysis. The remaining 62 cells were excluded.

The 2,638 retained cells were split into training and evaluation sets using an 80/20 split with random seed 123.

Gene filtering and selection were performed using the training observations only. Genes detected in at least three training cells were retained as candidates. From these genes, the 2,000 genes with the highest raw-count variance in the training data were selected as analysis features.

The same 2,000 selected genes were then retained for the evaluation
observations.

The `sample_id` column is an identifier containing the original cell barcode and should not be used as an analysis feature.

The `cell_type` column contains the Seurat cell-type annotation and should be treated as a label rather than an analysis feature. Cell-type labels may be used for post-hoc visualization and qualitative interpretation, but should not be used for preprocessing, dimension-reduction fitting, hyperparameter selection, or quantitative embedding evaluation.

No scaling or other feature preprocessing was applied when constructing this dataset. The gene-expression features remain raw nonnegative counts. The autonomous analysis should inspect the supplied representation before determining appropriate preprocessing.

## Cell types

- B
- CD14+ Mono
- CD8 T
- DC
- FCGR3A+ Mono
- Memory CD4 T
- Naive CD4 T
- NK
- Platelet
