
---
name: preprocessing
description: Choose model-compatible preprocessing without train/evaluation leakage.
---

# Preprocessing

## Method

Use `data_profile.json`, the dataset schema, user objective, and selected model to choose preprocessing. Available methods are `none`, `standard`, and `log_normalize`.

Pass the selected method to the training script using `--preprocessing`. Do not select preprocessing solely from the dataset name.

- Images: preserve channel order and use model-specific pixel scaling.
- scRNA-seq: distinguish raw counts from normalized expression. Preserve raw counts for count likelihoods and use transformations when justified.
- Exclude identifiers; retain feature ordering and relevant label metadata.

Check which preprocessing methods each model supports. Do not silently substitute a different method.

## Train and Evaluation Data

Fit learned transformations on training data only and apply them consistently to evaluation data. Do not silently drop observations or features.

## Output

Record the selected preprocessing, rationale, and settings in the model configuration and final report. Check shapes, finite values, valid ranges, and split integrity.

