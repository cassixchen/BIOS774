# Data Inspection Skill

## Purpose

Use this skill before selecting preprocessing or dimension-reduction methods for a new dataset.

Inspect the dataset so that later analysis decisions are based on observed data characteristics, feature meaning, and the analysis objective.

## Procedure

1. Read `<DATA_DIR>/DATA.md` to understand the dataset and analysis objective.

2. Read `<DATA_DIR>/schema.json` to identify label and identifier columns.

3. Run:

   ```bash
   python3 analysis/inspect_data.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR>
   ```

4. Inspect:

   `<OUTPUT_DIR>/data_profile.json`

5. Use the dataset documentation and resulting profile when selecting preprocessing and dimension-reduction methods.

## Properties to Inspect

Pay attention to:

- number of samples and features;
- numeric versus categorical features;
- missing-value count and fraction;
- zero count and fraction;
- feature means, standard deviations, minima, and maxima;
- row-total distribution when available;
- identifier and label columns.

## Interpretation

Use the dataset characteristics, feature meaning, and analysis objective together when making analysis decisions.

Consider standardization when feature scales differ substantially.

High zero fraction may indicate sparsity but does not alone determine preprocessing.

For nonnegative count data, varying row totals may support count normalization.

Address missing values before applying methods that cannot handle them, and do not automatically use categorical features as numeric inputs.

Exclude labels and identifiers from dimension reduction. Labels may be used for post-hoc visualization or interpretation.

## Reporting

Report only the dataset characteristics relevant to preprocessing, method selection, hyperparameter selection, evaluation, or interpretation.