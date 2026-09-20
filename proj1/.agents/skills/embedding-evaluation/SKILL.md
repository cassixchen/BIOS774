# Embedding Evaluation Skill

## Purpose

Use this skill after running one or more dimension-reduction methods.

Evaluate the resulting embeddings quantitatively and qualitatively to determine how well they represent the structure of the original data and to identify important limitations or sensitivity.

## Procedure

1. Identify the dimension-reduction methods that were run.

2. Run:

   ```bash
   python3 analysis/evaluate_embeddings.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR>
   ```

3. Inspect:

   `<OUTPUT_DIR>/embedding_evaluation.json`

4. Inspect the metrics and plots produced by each selected dimension-reduction method.

5. Compare the results in the context of the analysis objective and the properties of each method.

6. If an important result appears sensitive to a method choice or hyperparameter, consider a limited follow-up analysis.

## Evaluation Options

### Quantitative Evaluation

Use trustworthiness as a common neighborhood-preservation metric across the selected methods.

Higher trustworthiness indicates that local neighborhoods in the embedding more closely agree with neighborhoods in the corresponding preprocessed feature space.

For methods that support out-of-sample transformation, consider both training and evaluation trustworthiness.

Also consider available method-specific metrics:

- PCA: explained variance;
- MDS: stress;
- Isomap: reconstruction error;
- LLE: reconstruction error;
- t-SNE: KL divergence.

Interpret method-specific metrics according to the method that produced them rather than comparing their numerical values directly across methods.

### Qualitative Evaluation

Inspect the generated two-dimensional plots for meaningful structure, overlap, separation, unusual observations, or other patterns relevant to the analysis objective.

When evaluation embeddings are available, compare their general structure with the training embedding.

If labels are available, they may be used for post-hoc visualization and interpretation but should not determine which unsupervised embedding is preferred solely because it produces visually cleaner label separation.

Do not infer clusters, populations, trajectories, or other scientific conclusions from visual separation alone.

## Interpretation

Do not select a preferred representation from a single metric alone.

Consider quantitative metrics, visual structure, the objective of the analysis, method assumptions, and important limitations together.

Training performance alone does not demonstrate that an embedding generalizes to new observations when evaluation results are available.

Differences between methods should be interpreted in terms of what each method is designed to preserve.

## Reporting

Report the most relevant quantitative and qualitative findings.

Explain what the evaluation supports, note important limitations, and describe any follow-up or sensitivity analysis performed.