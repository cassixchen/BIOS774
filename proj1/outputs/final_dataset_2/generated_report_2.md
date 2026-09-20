# Dataset 2: exploratory dimension-reduction analysis

## Main findings

Log-normalized gene expression shows broad structure in both PCA and UMAP, with similar broad regions in held-out cells. Two-dimensional PCA retains only **10.03%** of training variance; **881 components** are required to reach 90%. UMAP makes regions look more separated but does not materially improve the reported local-neighborhood preservation over PCA. These are exploratory representations, not validated cell populations or trajectories. Numerical warnings during projection and evaluation remain an important qualification.

## Data and inspection

Input: `data/dataset_2`. All analysis outputs are in `outputs/final_dataset_2`. The analysis used the supplied repository tools and statistical skills, without modifying the inputs or creating additional analysis scripts.

The documentation describes PBMC3K single-cell expression counts, originally 2,700 cells and 32,738 genes. The supplied split contains 2,160 training cells and 540 evaluation cells (80/20, split seed 42). Gene selection was performed on training cells only: genes detected in at least three training cells were candidates, and the 2,000 highest raw-count-variance genes were retained in both splits. This selection is already part of the supplied dataset, not a new analysis result.

The schema identifies `sample_id` as a barcode identifier; it was excluded from features. There are no supplied labels, outcomes, or cell-type annotations.

The training profile reports:

| Property | Observed value |
|---|---:|
| Training cells | 2,160 |
| Numeric gene features | 2,000 |
| Categorical features | 0 |
| Missing feature values | 0 |
| Zero feature values | 3,200,262 (74.0801%) |
| Minimum cell total | 392 |
| Median cell total | 1,862 |
| Mean cell total | 2,008.2347 |
| Maximum cell total | 12,562 |

Counts and feature scales vary substantially. For example, FTL has mean 28.3028, standard deviation 45.8667, and maximum 391; EZR has mean 0.2333, standard deviation 0.5343, and maximum 5. These examples illustrate scale differences, not evidence that either gene drives an embedding axis. The inspection tool profiles training data only; it does not produce a separate evaluation-data quality profile.

## Preprocessing and method selection

Every run explicitly used `--preprocessing log_normalize`: divide each cell's counts by its total across the **2,000 retained genes**, multiply by 10,000, and apply the natural-log transformation `log(1 + x)`. The raw-count documentation and approximately 32-fold range in observed cell totals support this choice. The log transformation reduces the influence of large counts while retaining zeros.

`none` would leave strong differences in cell totals and large counts untreated. `standard` alone would equalize gene variances without correcting cell totals and could give sparse, low-count genes undue influence. No additional gene standardization, imputation, cell removal, or feature selection was performed. PCA centers the transformed features internally. Normalization uses each cell's own total and a fixed target, so it estimates no shared parameters from evaluation cells. Both fitted dimension-reduction models used training cells only and then transformed evaluation cells.

| Method | Purpose | Settings |
|---|---|---|
| PCA | Linear baseline and assessment of variance retained | All available components fitted with the supplied `PCA()` implementation; first two saved |
| UMAP | Complementary nonlinear view of local neighborhoods with evaluation transformation | 2 dimensions; 15 neighbors; minimum distance 0.1; Euclidean metric; random seed 123 |

UMAP used all 2,000 log-normalized features directly, not the two PCA coordinates. Its initial neighborhood size is the supplied default, a local scale relative to 2,160 training cells. Euclidean distance also matches the evaluator's reference distance. Minimum distance 0.1 and seed 123 were retained without tuning for visual separation. PCA's supplied script does not expose a random-seed argument.

Other available methods were considered but not run:

- **Kernel PCA:** no specific kernel or bandwidth is justified by the documentation; it would add a similarity assumption without ordinary PCA's variance interpretation.
- **MDS:** global pairwise-distance reproduction is not the principal objective; its optimization adds cost and the supplied implementation cannot transform evaluation cells.
- **Isomap:** there is no established connected manifold whose geodesic distances should be preserved; sparse, noisy counts make that assumption uncertain.
- **LLE:** local linear reconstruction is not established for these data.
- **Laplacian Eigenmaps:** overlaps with the neighborhood objective of UMAP but lacks evaluation transformation in this workflow.
- **t-SNE:** offers another local visualization but cannot transform evaluation cells in the supplied workflow; UMAP already addresses the local visualization objective.

## Quantitative evaluation

The supplied evaluator computed trustworthiness at **10 neighbors**, separately within training and evaluation sets, using the same log-normalized feature representation for every method. Higher values indicate fewer strongly misplaced neighbors in the embedding. Trustworthiness is not a percentage of correctly recovered neighbors and does not measure global distance fidelity or cluster validity.

| Representation | Training trustworthiness | Evaluation trustworthiness |
|---|---:|---:|
| PCA, first two components | 0.788951 | 0.785040 |
| UMAP, 15 neighbors | 0.789350 | 0.780356 |
| UMAP, 30 neighbors (follow-up) | 0.787554 | 0.781780 |

The scores offer no compelling advantage for the visually more separated UMAP map. PCA and UMAP serve complementary purposes; neither is selected as an overall winner. Evaluation scores use neighbors among the 540 evaluation cells, not evaluation-to-training neighbor matching. Different sample sizes and neighborhood scales limit direct interpretation of train–evaluation differences. No confidence intervals or significance tests were produced.

PCA explains **7.4022%** of variance with PC1 and **2.6322%** with PC2, totaling **10.0345%**. Reaching 90% requires **881 components**. Thus the two-dimensional plot captures major directions but omits most feature variation; it should not substitute for a richer representation in downstream analysis.

## Visual interpretation

The training PCA plot shows a dense region at negative PC1 with an upward extension, another lower region at negative PC2, and an elongated region at positive PC1. Sparse observations lie between these regions. Held-out cells occupy broadly similar areas under the training-fitted projection. These observations describe geometry only; no gene loadings or cell identities were inferred.

The 15-neighbor UMAP training plot shows a large elongated region with a smaller attached lobe, a clearly separated compact region, and an isolated plotted point. Evaluation cells occupy corresponding broad locations but form conspicuous hollow outlines rather than filling the training regions. This train–transform discrepancy limits density interpretation: the outlines do not establish biological rings, trajectories, or empty states. Likewise, distances between UMAP regions are not calibrated expression distances, and the isolated point is not evidence of a rare cell type.

Training and evaluation plots use independently chosen axis limits. Visual density is also affected by unequal sample sizes and point overlap. There was no formal observation-level correspondence analysis between PCA and UMAP regions.

## Targeted follow-up

The separated UMAP regions, isolated plotted point, and hollow evaluation outlines motivated one sensitivity check: increase `n_neighbors` from 15 to 30, keeping preprocessing, dimensionality, minimum distance, metric, and seed fixed. Outputs were saved separately in `followup_umap_n30`, and the supplied evaluator was rerun there at 10 neighbors.

At 30 neighbors, training trustworthiness is **0.787554** and evaluation trustworthiness is **0.781780**. The broad visual arrangement remains: one large region with a smaller lobe, a separated region, and an isolated plotted point. The large region changes orientation and shape, while hollow evaluation outlines persist. This supports only qualitative stability of broad appearance across these two neighborhood settings; it does not establish stable cell membership or quantify alignment between runs. Increasing the neighborhood size did not resolve the train–evaluation density discrepancy or meaningfully alter the overall evaluation conclusion. The follow-up evaluator emitted the same matrix-multiplication warnings.

No additional seed sweep, metric search, or tuning for clearer separation was performed. The 15-neighbor run remains the documented baseline rather than replacing it based on a small score difference.

## Warnings and limitations

Initial PCA and UMAP invocations terminated with exit status 134 before completing plots. Rerunning the unmodified tools with `MPLBACKEND=Agg` completed successfully. The completed runs also set `VECLIB_MAXIMUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`. Matplotlib reported an unwritable default cache and used temporary caches; this was an environment warning rather than a reported data error.

PCA evaluation transformation and the trustworthiness evaluator emitted divide-by-zero, overflow, and invalid-value warnings in matrix multiplication. Single-threaded reruns did **not** eliminate them. The saved baseline embeddings have the expected row counts and no textual NaN, infinity, or empty coordinate fields; plots and finite evaluation scores were produced. However, successful output alone does not prove numerical accuracy. The cause remains unresolved within the supplied-tool workflow, so exact projection and trustworthiness values should be treated as provisional rather than fully numerically validated. Warnings were not suppressed. UMAP also warned that fixing the random seed forces single-thread execution; that warning concerns reproducibility and speed.

Normalization totals refer only to retained genes, not complete transcriptome library sizes. Selection by raw-count variance can favor abundant or depth-associated genes, and this analysis cannot recover information from omitted genes. Sparse counts, technical variation, and composition effects may remain after normalization. The supplied tools do not provide marker-gene testing, PCA loading export, doublet detection, mitochondrial-quality assessment, batch adjustment, or quantitative embedding-stability analysis; none was implemented during this run. No cell types, cluster counts, differential expression, or biological trajectories are established.

The evaluation split comes from the same source dataset and supports only within-dataset assessment, not generalization to new donors, batches, or studies.

## Output guide

All six generated plots were visually inspected:

| Analysis | Training plot | Evaluation plot |
|---|---|---|
| PCA | [Training PCA](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/pca_train_2d.png) | [Evaluation PCA](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/pca_eval_2d.png) |
| UMAP, 15 neighbors | [Training UMAP](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/umap_train_2d.png) | [Evaluation UMAP](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/umap_eval_2d.png) |
| UMAP, 30 neighbors | [Training follow-up](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/followup_umap_n30/umap_train_2d.png) | [Evaluation follow-up](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/followup_umap_n30/umap_eval_2d.png) |

The [data profile](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/data_profile.json) contains the training summaries. [PCA metrics](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/pca_metrics.json) contain the complete explained-variance spectrum. [Baseline evaluation](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/embedding_evaluation.json) and [follow-up evaluation](/Users/cassixchen/Desktop/unc/BIOS774/proj1/outputs/final_dataset_2/followup_umap_n30/embedding_evaluation.json) contain the reported trustworthiness values. Each UMAP directory has its settings in `umap_metrics.json`.

Embedding CSVs contain 2,160 training rows or 540 evaluation rows, plus a header, and two coordinates per observation. The supplied tools omit identifiers from these files; rows correspond to the original input row order and must be joined to barcodes by that order. The tools save settings and coordinates but do not serialize fitted model objects. No unsupported biological identities were assigned.
