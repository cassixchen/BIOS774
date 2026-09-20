# Laplacian Eigenmaps Analysis Skill

## Purpose

Use this skill when Laplacian Eigenmaps is appropriate for representing local nonlinear structure in the data.

Laplacian Eigenmaps constructs a neighborhood graph and uses its spectral structure to produce a lower-dimensional representation that preserves local relationships.

## When to Consider Laplacian Eigenmaps

Consider Laplacian Eigenmaps when:

- local neighborhood connectivity is the main structure of interest;
- a graph-based representation of the observations is appropriate;
- preserving nearby relationships is more important than preserving global distances.

Laplacian Eigenmaps may be less appropriate when global geometry or out-of-sample transformation is important.

Compared with LLE, this method is motivated by the neighborhood graph and its spectral structure rather than local linear reconstruction.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Select an appropriate neighborhood size.

3. Run:

   ```bash
   python3 analysis/run_laplacian_eigenmaps.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy> \
     --n-neighbors <n_neighbors>
   ```

4. Inspect:

   - `laplacian_eigenmaps_metrics.json`
   - `laplacian_eigenmaps_train.csv`
   - `laplacian_eigenmaps_train_2d.png`

5. Use the Laplacian Eigenmaps results together with embedding evaluation and visualization to determine whether the representation is useful.

## Hyperparameters

`n_neighbors` controls the neighborhood graph used to construct the embedding.

Smaller values emphasize more local relationships but may produce a poorly connected or unstable graph. Larger values produce a more connected graph but may weaken the emphasis on local structure.

Choose the neighborhood size based on the data characteristics and analysis objective rather than assuming the default is optimal.

If the result appears sensitive to neighborhood size, consider a limited sensitivity analysis using another reasonable value.

## Laplacian Eigenmaps Output

The implementation produces a two-dimensional embedding of the training data.

The current implementation does not provide an out-of-sample transformation for evaluation data.

Pay attention to:

- training trustworthiness;
- structure visible in the two-dimensional embedding;
- sensitivity to neighborhood size when relevant.

## Interpretation

Interpret Laplacian Eigenmaps as a graph-based representation that emphasizes local neighborhood structure.

Do not treat distances or separation in the embedding as direct measurements of global distance in the original feature space.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting the embedding.

## Reporting

Report why Laplacian Eigenmaps was selected, the preprocessing and neighborhood size used, and the most relevant evaluation results.

Mention the lack of out-of-sample transformation and other important limitations when they affect interpretation.