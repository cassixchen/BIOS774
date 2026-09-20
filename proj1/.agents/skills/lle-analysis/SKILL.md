# LLE Analysis Skill

## Purpose

Use this skill when LLE is appropriate for representing nonlinear structure that can be described by local relationships between nearby observations.

LLE constructs a lower-dimensional representation by preserving local neighborhood relationships.

## When to Consider LLE

Consider LLE when:

- nonlinear manifold structure is plausible;
- observations can reasonably be approximated by linear combinations of nearby observations;
- preserving these local reconstruction relationships is relevant to the analysis objective.

LLE may be less appropriate when global distances are important, local neighborhoods are unstable, or there is little reason to expect the manifold to be locally linear.

Compared with other neighborhood methods, prefer LLE when the local linear reconstruction assumption is meaningful rather than simply because local structure is of interest.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Select an appropriate neighborhood size.

3. Run:

   ```bash
   python3 analysis/run_lle.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy> \
     --n-neighbors <n_neighbors>
   ```

4. Inspect:

   - `lle_metrics.json`
   - `lle_train.csv`
   - `lle_eval.csv`
   - `lle_train_2d.png`
   - `lle_eval_2d.png`

5. Use the LLE results together with embedding evaluation and visualization to determine whether the representation is useful.

## Hyperparameters

`n_neighbors` controls the local neighborhoods used to construct the embedding.

Smaller values emphasize more local structure but may produce unstable neighborhoods. Larger values use broader neighborhoods but may weaken the assumption that the manifold is locally linear.

Choose the neighborhood size based on the data characteristics and analysis objective rather than assuming the default is optimal.

If the result appears sensitive to neighborhood size, consider a limited sensitivity analysis using another reasonable value.

## LLE Output

The implementation fits LLE using the training data and applies the fitted transformation to evaluation data.

Pay attention to:

- train and evaluation trustworthiness;
- reconstruction error;
- structure visible in the two-dimensional embeddings;
- sensitivity to neighborhood size when relevant.

## Interpretation

Interpret LLE as a nonlinear representation that attempts to preserve local neighborhood relationships.

Do not assume that global distances in the embedding represent global distances in the original data.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting LLE.

## Reporting

Report why LLE was selected, the preprocessing and neighborhood size used, and the most relevant evaluation results.

Mention important sensitivity or limitations when they affect interpretation.