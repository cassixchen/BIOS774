# t-SNE Analysis Skill

## Purpose

Use this skill when t-SNE is appropriate for visualizing local structure in high-dimensional data.

t-SNE produces a nonlinear embedding that emphasizes preservation of local neighborhood relationships.

## When to Consider t-SNE

Consider t-SNE when:

- exploratory visualization is the primary objective;
- very local neighborhood structure is of interest;
- nonlinear patterns may not be represented well by linear methods.

t-SNE may be less appropriate when global distances or out-of-sample transformation are important.

Compared with methods intended to preserve broader geometry, prefer t-SNE when the main goal is exploratory visualization of local neighborhood structure.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Select appropriate t-SNE hyperparameters.

3. Run:

   ```bash
   python3 analysis/run_tsne.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy> \
     --perplexity <perplexity> \
     --learning-rate <learning_rate> \
     --init <init> \
     --random-state <random_state>
   ```

4. Inspect:

   - `tsne_metrics.json`
   - `tsne_train.csv`
   - `tsne_train_2d.png`

5. Use the t-SNE results together with embedding evaluation and visualization to determine whether the representation is useful.

## Hyperparameters

`perplexity` affects the neighborhood scale considered by t-SNE. Choose a value appropriate for the number of observations and the local structure of interest.

`learning_rate` affects the optimization used to construct the embedding.

`init` determines how the embedding is initialized.

Use `random_state` to make the analysis reproducible.

Choose hyperparameters based on the data characteristics and analysis objective rather than assuming the defaults are optimal.

If the conclusions appear sensitive to important hyperparameters, consider a limited sensitivity analysis.

## t-SNE Output

The implementation produces a two-dimensional embedding of the training data.

The current implementation does not provide an out-of-sample transformation for evaluation data.

Pay attention to:

- training trustworthiness;
- KL divergence;
- structure visible in the two-dimensional embedding;
- sensitivity to important hyperparameters when relevant.

## Interpretation

Interpret t-SNE primarily as a visualization of local neighborhood structure.

Do not interpret distances between separated groups or their relative positions as reliable measures of global relationships.

Visible clusters should not automatically be interpreted as distinct populations without additional evidence.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting t-SNE.

## Reporting

Report why t-SNE was selected, the preprocessing and hyperparameters used, and the most relevant evaluation results.

Mention the lack of out-of-sample transformation and other important limitations when they affect interpretation.