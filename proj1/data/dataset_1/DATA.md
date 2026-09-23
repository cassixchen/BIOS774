# Dataset 1: PathMNIST

This dataset is a stratified subset of PathMNIST from the MedMNIST collection.

Each observation is a 28 x 28 RGB histopathology image flattened into 2,352 pixel features.

Training observations: 4500

Evaluation observations: 900

The training observations were sampled from the original PathMNIST training split. Evaluation observations were sampled from the original PathMNIST test split.

The subset contains an equal number of observations from each tissue class.

The `sample_id` column is an identifier and should not be used as an analysis feature.

The `tissue_type` column contains the PathMNIST tissue-class name and should be treated as a label rather than an analysis feature. Tissue-type labels may be used for post-hoc visualization and qualitative interpretation, but should not be used for preprocessing, dimension-reduction fitting, hyperparameter selection, or quantitative embedding evaluation.

No scaling or other feature preprocessing was applied when constructing this dataset. The autonomous analysis should inspect the supplied representation before determining appropriate preprocessing.

## Tissue types

- adipose
- background
- debris
- lymphocytes
- mucus
- smooth muscle
- normal colon mucosa
- cancer-associated stroma
- colorectal adenocarcinoma epithelium
