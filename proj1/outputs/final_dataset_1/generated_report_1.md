# Dataset 1: exploratory dimension-reduction analysis

## Findings

PCA and UMAP reveal broad structure in the raw image pixels, with considerable overlap between tissue labels. The first two principal components explain **57.01%** of training variance; **161 components** are required for 90%. UMAP provides a complementary local-neighborhood view, but its trustworthiness is similar to PCA's. Increasing UMAP's neighborhood size from 15 to 30 preserves the broad visual pattern while changing the arrangement of small isolated regions. These findings do not establish discrete tissue clusters or classification performance. Numerical warnings remain unresolved, so the quantitative comparisons below are provisional.

## Data and preprocessing

The input was `data/dataset_1`, interpreted using its `DATA.md` and `schema.json`. The documentation identifies a stratified PathMNIST subset: flattened 28 × 28 RGB histopathology images, sampled from the original training and test splits. The tools confirm 4,500 training observations, 900 evaluation observations, and 2,352 numeric features. The documentation describes equal representation of nine tissue classes; class counts were not independently tabulated by the supplied inspection tool.

The schema excludes `sample_id` and `tissue_type` from features. Labels were used only to color plots and interpret results after fitting. No label-based tuning was performed.

The training profile reports:

| Property | Observed value |
|---|---:|
| Missing feature values | 0 |
| Categorical feature columns | 0 |
| Pixel range across features | 0–255 |
| Feature standard-deviation range | 30.9005–47.1394 |
| Zero values | 1,804 (0.01704%) |
| Minimum / median / maximum image pixel sum | 171,239 / 395,399 / 564,990 |

**Preprocessing: `none`**, explicitly passed to every run. Pixels share the same intensity units and comparable variability. Retaining their original scale preserves intensity differences without upweighting less-variable pixel locations. Standardization was not necessary on this evidence. Log normalization was inappropriate because these are image intensities, not counts with differing sampling depth; differences in pixel sums do not by themselves justify normalizing image brightness away. PCA still performs its own training-based mean centering.

The inspection tool profiles training data only. A separate evaluation missingness or distribution profile was not produced. Both selected models were fitted exclusively on training features, then used to transform evaluation features into the same fitted coordinate system.

## Methods and settings

The repository's data-inspection, preprocessing, PCA, UMAP, and embedding-evaluation skills guided the analysis. The other method skills were reviewed to assess alternatives. All computation used the supplied scripts and repository `.venv/bin/python3`; no new analysis scripts or dependencies were added.

| Method | Purpose | Settings |
|---|---|---|
| PCA | Global linear baseline and variance summary for correlated pixels | All components computed internally; first two saved; `preprocessing=none`; script's default PCA solver |
| UMAP | Nonlinear neighborhood visualization with evaluation transformation | 2 dimensions; 15 neighbors; `min_dist=0.1`; Euclidean metric; seed 123; `preprocessing=none` |
| UMAP follow-up | Check sensitivity of isolated regions to neighborhood scale | 30 neighbors; all other settings unchanged |

The initial UMAP settings are reasonable local-exploration defaults, not claimed optima. Euclidean distance provides a direct common pixel-space reference for PCA and UMAP. No PCA preprocessing stage was added before UMAP because the supplied UMAP tool operates directly on the selected feature representation. PCA exposes no random-seed option and computes the complete component spectrum.

Methods not selected:

- **Kernel PCA:** no particular kernel similarity or bandwidth was motivated by the documentation; it would add tuning and a dense kernel calculation without ordinary PCA's variance interpretation.
- **MDS:** global pairwise-distance optimization was not the main objective, adds computational cost, and the current implementation cannot transform evaluation images.
- **Isomap:** the documentation does not establish a manifold whose graph geodesic distances would meaningfully describe these raw pixels.
- **LLE:** local linear reconstruction of neighboring raw images was not a justified assumption for this task.
- **Laplacian Eigenmaps:** overlaps with UMAP's local-graph purpose and lacks evaluation transformation in this workflow.
- **t-SNE:** useful for local visualization, but its training-only implementation would not support the shared training/evaluation representation prioritized here.

## Quantitative evaluation

The supplied evaluator computed trustworthiness with **10 neighbors**, separately within training and evaluation observations, using the untransformed Euclidean pixel representation for all runs. Higher scores indicate fewer intruding neighbors in the two-dimensional representation; they do not measure tissue-label accuracy, global distance preservation, or all forms of neighborhood distortion.

| Representation | Training trustworthiness | Evaluation trustworthiness |
|---|---:|---:|
| PCA, 2 dimensions | 0.805881 | 0.847546 |
| UMAP, 15 neighbors | 0.814062 | 0.846841 |
| UMAP, 30 neighbors | 0.811855 | 0.847410 |

PCA explains 54.7984% of training variance in PC1 and 2.2123% in PC2, totaling 57.0107%. Thus the dominant linear direction is strong, but two coordinates omit approximately 42.99% of variance. The need for 161 components to reach 90% argues against treating the images as adequately summarized by two linear dimensions. Explained variance was computed on training data; evaluation explained variance was not produced.

UMAP's slightly higher training trustworthiness is insufficient to declare it superior. Evaluation scores are nearly identical. Training and evaluation scores use different sample sizes and within-split neighborhoods, so higher evaluation scores do not establish better generalization. No uncertainty estimates or statistical significance tests were produced.

## Visual interpretation and sensitivity

All six generated embedding plots were inspected. **Colors change between training and evaluation plots because the plotting tool assigns them in label-appearance order. Compare legend label numbers, not colors.** Label meanings from the documentation are: 0 adipose, 1 background, 2 debris, 3 lymphocytes, 4 mucus, 5 smooth muscle, 6 normal colon mucosa, 7 cancer-associated stroma, and 8 colorectal adenocarcinoma epithelium.

**PCA:** the training plot has a broad overlapping central cloud, an adipose-rich region toward positive PC1, and a background-rich region at negative PC1 and negative PC2. The evaluation plot retains these broad features. Debris includes a more prominent high-PC2 extension in evaluation, while several other tissue types overlap centrally. This is a descriptive split difference, not a formal distribution-shift result. The supplied outputs do not include loadings or representative images, so PC1 cannot be definitively identified as brightness or a particular tissue property.

**UMAP, 15 neighbors:** most observations form a connected-looking elongated body with extensive label mixing. Adipose is concentrated toward the upper end. Background-rich groups appear apart from much of the main body, and there are small satellite regions. Evaluation observations broadly occupy corresponding areas of the fitted embedding, while tissue labels remain mixed. The apparent branches, gaps, and separations are not evidence of biological trajectories or distinct populations.

**Targeted follow-up:** the satellites and gaps motivated one increase to 30 neighbors, saved separately in `followup_umap_n30`. The broad adipose-rich end, background-rich separated region, and mixed central body persist. The lower background-rich region appears closer to the main body, and a small debris-rich satellite changes position substantially. Trustworthiness remains similar, as shown above. This supports the broad descriptive pattern across these two neighborhood choices, but not the exact placement or spacing of islands. Only neighborhood size was varied; seed and `min_dist` sensitivity were not tested. The original 15-neighbor result is retained rather than selecting whichever plot appears most separated.

PCA is useful for summarizing global pixel variation and its compression cost; UMAP is useful for inspecting local relationships. Neither yields a clean nine-class partition, and neither is a validated classifier.

## Warnings and limits

Initial PCA and UMAP processes exited with status 134 after writing coordinates and metrics but before completing plots. Rerunning the same supplied tools with `MPLBACKEND=Agg` completed successfully and produced all requested plots. The successful runs also set `VECLIB_MAXIMUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Initial partial results were replaced by these completed runs.

PCA transformation emitted divide-by-zero, overflow, and invalid-value warnings in matrix multiplication. Similar warnings occurred in the evaluator's matrix multiplication for both the main and follow-up evaluations. Single-thread settings did not eliminate them. The CSVs had the expected row counts, a text check found no literal NaN or infinity tokens in the main embeddings, plots rendered, and the supplied evaluator accepted the coordinates and returned finite scores. These checks are reassuring but **do not establish numerical correctness or explain the warnings**. The underlying numerical-library issue remains unresolved; the reported trustworthiness scores and affected transformations should be treated as provisional. The supplied tools do not provide an alternate numerical implementation for an independent cross-check, and no custom validation calculation was introduced under repository restrictions.

Matplotlib used temporary font caches because its default cache directory was not writable. UMAP warned that a fixed seed forces single-thread execution; this is expected and preserves reproducibility. An initial aborted UMAP process also reported a leaked semaphore at shutdown; the successful rerun completed without that shutdown warning.

Raw-pixel distances are sensitive to brightness, color, and image alignment and are not established measures of tissue morphology. No image-aware feature extraction, stain adjustment, representative-image inspection, duplicate assessment, or patient-level analysis was supplied by these tools. The balanced subset is not evidence of real-world tissue prevalence. Plot overlap can conceal observations, and UMAP spacing between distant groups should not be interpreted quantitatively. These results remain exploratory descriptions of the supplied pixel representation.

## Output record

All analysis artifacts are under `outputs/final_dataset_1`:

- `data_profile.json`: training feature inspection.
- `pca_metrics.json`: preprocessing, sample counts, full variance spectrum and 90% threshold.
- `pca_train.csv`, `pca_eval.csv`, `pca_train_2d.png`, `pca_eval_2d.png`: PCA coordinates and plots.
- `umap_metrics.json`, `umap_train.csv`, `umap_eval.csv`, `umap_train_2d.png`, `umap_eval_2d.png`: primary UMAP settings, coordinates and plots.
- `embedding_evaluation.json`: primary quantitative evaluation.
- `followup_umap_n30/`: UMAP settings, two coordinate files, two plots, and `embedding_evaluation.json` for the targeted follow-up.
- `generated_report_1.md`: this report.

Coordinate CSVs preserve input row order but do not contain sample identifiers; associate them with the corresponding original split by row order. Inputs were not modified. Models themselves are not serialized by the supplied tools.
