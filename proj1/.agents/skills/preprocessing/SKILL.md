# Preprocessing Skill

## Purpose

Use this skill after data inspection and before dimension reduction.

Choose preprocessing based on the observed data characteristics, feature meaning, analysis objective, and requirements of the selected dimension-reduction method.

## Procedure

1. Read `<DATA_DIR>/DATA.md`.

2. Read `<DATA_DIR>/schema.json`.

3. Inspect `<OUTPUT_DIR>/data_profile.json`.

4. Select one of the available preprocessing strategies:

   - `none`
   - `standard`
   - `log_normalize`

5. Explicitly pass the selected strategy when running a dimension-reduction method:

   ```bash
   --preprocessing <strategy>
   ```

6. Use the same preprocessing strategy when evaluating the resulting embedding.

## Preprocessing Options

### No Transformation

Consider `none` when features are already on comparable scales or preserving their original scale is appropriate.

Do not apply a transformation simply because one is available.

### Standardization

Consider `standard` when features have substantially different scales and those differences should not determine the dimension-reduction result.

Standardization may be particularly relevant for variance-, distance-, and neighborhood-based methods.

The implementation uses `StandardScaler`, fitted on training data only and then applied to evaluation data.

### Log Normalization

Consider `log_normalize` when features are nonnegative, count-like measurements and observations may have different total counts.

The implementation normalizes each observation to a total count of 10,000 and then applies `log(1 + x)`.

Use the dataset documentation and inspection results together to determine whether count normalization is appropriate.

Do not use `log_normalize` simply because the data are nonnegative, sparse, or skewed.

## Data Requirements

Identifier and label columns must not be used as input features for unsupervised dimension reduction.

Do not pass raw categorical variables into methods requiring numeric input.

The current dimension-reduction tools require finite numeric input. If the available preprocessing cannot appropriately handle the data, report the limitation rather than silently modifying the data.

## Reporting

Report the selected preprocessing strategy and the evidence supporting the decision.

Mention important alternatives or limitations when they affect interpretation.