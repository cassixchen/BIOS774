# Dataset 1: PathMNIST

This dataset is a stratified subset of PathMNIST from the MedMNIST collection.

Each observation is a 28 x 28 RGB histopathology image flattened into 2,352 pixel features.

Training observations: 4500

Evaluation observations: 900

The training observations were sampled from the original PathMNIST training split. Evaluation observations were sampled from the original PathMNIST test split.

The subset contains an equal number of observations from each tissue class.

The `sample_id` column is an identifier and should not be used as an analysis feature.

The `tissue_type` column contains the PathMNIST class label and should not be used as an input feature for unsupervised dimension reduction. It may be used after fitting for visualization and interpretation.

No scaling or other feature preprocessing was applied when constructing this dataset.

## Tissue labels

- 0: adipose
- 1: background
- 2: debris
- 3: lymphocytes
- 4: mucus
- 5: smooth muscle
- 6: normal colon mucosa
- 7: cancer-associated stroma
- 8: colorectal adenocarcinoma epithelium
