# MDS Analysis Skill

## Purpose

Use this skill when MDS is appropriate for producing a lower-dimensional representation that preserves pairwise distances between observations.

MDS represents observations so that distances in the embedding approximate distances in the original feature space.

## When to Consider MDS

Consider MDS when:

- preserving pairwise distances is central to the analysis objective;
- the relative geometry of observations is more important than explaining feature variance;
- the dataset is small enough for pairwise-distance calculations and optimization to be practical.

MDS may be less appropriate for large datasets because of its computational cost.

Compared with neighborhood-based methods, prefer MDS when preserving overall pairwise distances rather than primarily local relationships is the goal.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Run:

   ```bash
   python3 analysis/run_mds.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy>
   ```

3. Inspect:

   - `mds_metrics.json`
   - `mds_train.csv`
   - `mds_train_2d.png`

4. Use the MDS results together with embedding evaluation and visualization to determine whether the representation is useful.

## MDS Output

The implementation produces a two-dimensional embedding of the training data.

The current implementation does not provide an out-of-sample transformation for evaluation data.

Pay attention to:

- training trustworthiness;
- stress;
- structure visible in the two-dimensional embedding.

Lower stress indicates that distances in the embedding more closely reproduce the distances targeted by MDS, but interpret stress together with the visualization and other evaluation results.

## Interpretation

Interpret MDS as a representation designed to preserve pairwise distances.

Do not assume that visible groups in the embedding represent distinct populations without additional evidence.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting MDS.

## Reporting

Report why MDS was selected, the preprocessing used, and the most relevant evaluation results.

Mention computational cost, lack of out-of-sample transformation, or other important limitations when they affect interpretation.