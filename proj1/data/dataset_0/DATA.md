# Dataset 0: Breast Cancer Wisconsin

This dataset is the Breast Cancer Wisconsin diagnostic dataset provided by
scikit-learn.

Each observation represents a breast mass described by 30 numeric features
computed from a digitized image of a fine needle aspirate of the breast mass.

Training observations: 455

Evaluation observations: 114

The observations were split into training and evaluation sets using an 80/20
stratified split with random seed 42. Stratification was performed using the
diagnosis label.

The `sample_id` column is an identifier and should not be used as an analysis
feature.

The `diagnosis` column contains the diagnostic class label and should not be
used as an input feature for unsupervised dimension reduction. It may be used
after fitting for visualization and interpretation.

No scaling or other feature preprocessing was applied when constructing this
dataset.

## Diagnosis labels

- 0: malignant
- 1: benign