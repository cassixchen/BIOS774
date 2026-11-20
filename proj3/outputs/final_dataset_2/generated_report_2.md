# Paired RNA–ATAC representation learning

## Objective and data

The objective was to learn representations capturing relationships between gene expression and chromatin accessibility in paired single cells. Exact paired-cell cross-modal retrieval is a measurable proxy for this objective, though it does not exhaust biological utility. The source was the 10x Genomics 10k PBMC Multiome filtered matrix, `data/dataset_2/filtered_feature_bc_matrix.h5`: 10,970 cells, 36,601 gene-expression features and 111,743 peaks, with 113,471,052 sparse nonzero entries. RNA and ATAC measurements share cell barcodes. No cell-type labels were available.

A 1,000-cell raw inspection sample had RNA library-size 10th/median/90th percentiles of 997.9/2,544/5,186.3 and median 1,374 detected genes. ATAC counts were 7,166.5/22,536/40,337.6 at the same percentiles, with median 8,722.5 detected peaks. Seed 42 produced disjoint train, validation and test splits of 8,776, 1,097 and 1,097 cells. Prepared-data checks recorded unique IDs, matching manifest order, finite numeric values, no missing values, identical ordered columns, and disjoint IDs. Each row preserves the original cell pairing. IDs were not model features.

## Preprocessing

RNA counts were normalized per cell to 10,000, log1p transformed, and the 2,000 highest-variance genes selected. The roughly 5.2-fold RNA library-size p90/p10 ratio motivated depth normalization; variance selection prioritizes variation within a tractable subset of 36,601 genes. ATAC was binarized, TF-IDF transformed, and the 12,000 most frequent peaks selected. This emphasizes accessibility presence and reduces count-depth influence given the roughly 5.6-fold ATAC count p90/p10 ratio, at the cost of count magnitude and rare peaks. Each modality was reduced to 64 SVD components, limiting memory and projection-model cost. Sparse matrices were retained through reduction; the full high-dimensional matrix was not densified.

A feasible alternative was log1p RNA with 4,000 detection-selected genes and 128 components, plus count TF-IDF ATAC with 20,000 variance-selected peaks and 128 components. It retains more features and count magnitude but costs more and may retain more depth effects. It was considered, not benchmarked. Feature rankings, IDF and SVD were fitted on training cells only; validation and test received frozen transforms. Model-side imputation and scaling were also fitted on training data only.

Randomized SVD attempts for both modalities emitted divide-by-zero, overflow and invalid-matmul warnings despite finite outputs. Those attempts were rejected; selected ARPACK fits recorded no runtime warnings and finite outputs. Their explained-variance sums were 0.2398 for RNA and 0.1926 for ATAC, so considerable variation is not retained. The cause of the rejected attempts' warnings was not established.

## Model selection and protocol

Three candidates were trained on the same 8,776 training cells and compared on the **same 1,097 validation cells**, using only the mean of RNA→ATAC and ATAC→RNA Recall@5. All used 40 epochs, batch size 128, latent dimension 32 and seed 42; recorded PyTorch device was `mps`. RNA is modality 1 and ATAC modality 2. Linear encoders map 64 inputs directly to 32 dimensions; MLP encoders use a 128-unit ReLU hidden layer. Contrastive training used normalized embeddings and symmetric paired InfoNCE with temperature 0.1. The self-supervised candidate used separate within-modality mean-squared-error autoencoder reconstruction, which does not explicitly align spaces.

| Candidate | Objective; RNA / ATAC encoders | Validation RNA→ATAC R@5 | Validation ATAC→RNA R@5 | Mean R@5 |
|---|---|---:|---:|---:|
| 1 | Contrastive; linear / linear | 0.08933 | 0.07475 | **0.08204** |
| 2 | Contrastive; MLP / MLP | 0.08569 | 0.06472 | 0.07521 |
| 3 | Self-supervised; MLP / linear | 0.00456 | 0.00365 | 0.00410 |

Random validation Recall@5 is 5/1,097 = 0.00456. Candidate 1 won the prespecified criterion; the higher-capacity contrastive alternative did not improve that criterion, and the reconstruction candidate has no explicit cross-modal alignment objective. Supervised learning was unsuitable without genuine labels; a deeper MLP was rejected on capacity and compute grounds. The winner's trained checkpoint was copied without retraining, then evaluated on the untouched test split. Test results were not used for selection. The SVD input features were fixed during neural training; the two linear projection encoders were trained, not frozen or foundation-model fine-tuned.

## Untouched test results

Representation quality on 1,097 pairs: mean paired cosine 0.54859, mean negative cosine 0.05061, and positive-minus-negative gap 0.49798. Estimated positive-versus-negative pairwise AUC was 0.90150; mean squared alignment distance was 0.90281. RNA and ATAC uniformity were -2.53487 and -2.39243, and effective ranks were 9.88991 and 9.19812 out of 32 dimensions. AUC and uniformity used 100,000 sampled pairs with seed 42. Lower alignment and uniformity and higher effective rank can be favorable, but these measures alone do not establish biological meaning; normalized-embedding alignment is related to positive cosine.

The following retrieval results rank all 1,097 candidates per query. Recall values are fractions; random Recall@K is K/1,097.

| Metric | RNA→ATAC | ATAC→RNA | Bidirectional mean | Random |
|---|---:|---:|---:|---:|
| Recall@1 | 0.03191 | 0.02097 | 0.02644 | 0.00091 |
| Recall@5 | 0.09754 | 0.08295 | 0.09025 | 0.00456 |
| Recall@10 | 0.15314 | 0.15406 | 0.15360 | 0.00912 |
| Recall@20 | 0.24430 | 0.24521 | 0.24476 | 0.01823 |
| Recall@50 | 0.45123 | 0.42297 | 0.43710 | 0.04558 |
| Recall@100 | 0.63263 | 0.62808 | 0.63036 | 0.09116 |
| MRR | 0.07750 | 0.06767 | 0.07259 | — |
| Median rank | 61 | 67 | 64 | — |

The bidirectional median-rank entry is the mean of the two directional medians. Retrieval exceeds random ranking, but exact matching remains difficult: mean Recall@5 is about 9%. Biologically similar nonmatching cells count as negatives during training and as incorrect matches in retrieval; these possible false negatives and the lack of cell-type labels limit interpretation. One dataset and one split do not establish generalization or clinical or biological validity.

## Artifacts and reproduction

The saved held-out learned representations are [`contrastive/eval_embeddings.csv`](contrastive/eval_embeddings.csv). Each row has `sample_id` and concatenated modality embeddings: the first 32 latent columns are RNA, the next 32 ATAC. Other artifacts include [`contrastive/model.pt`](contrastive/model.pt), [`contrastive/preprocessing.joblib`](contrastive/preprocessing.joblib), [`prepared/feature_pipeline.joblib`](prepared/feature_pipeline.joblib), training history and configuration, [`embedding_quality_metrics.json`](contrastive/embedding_quality_metrics.json), [`downstream_quality_metrics.json`](contrastive/downstream_quality_metrics.json), the separate combined [`evaluation_metrics.json`](contrastive/evaluation_metrics.json), and per-pair ranks in `contrastive/expanded_retrieval/paired_cell_ranks.csv`. Candidate validation results and split information are in `model_selection_results.json` and `validation_split.json`.

The recorded preparation command is below. Run from the repository root using a **fresh** output path; the existing run must not be overwritten. The remaining commands reproduce the candidate workflow in a fresh directory, using the prepared validation split for selection and the copied winner for test. Device choice follows the host environment and exact numerical results can vary across devices and software versions.

```bash
mkdir -p outputs/REPRO
python analysis/inspect_raw.py --kind multiome --source data/dataset_2 --output outputs/REPRO/raw_profile.json
python analysis/prepare_dataset_2.py --source data/dataset_2 --output outputs/REPRO/prepared --train-size 8776 --validation-size 1097 --eval-size 1097 --rna-features 2000 --atac-features 12000 --rna-components 64 --atac-components 64 --rna-normalization library-log1p --rna-feature-selection variance --atac-feature-selection frequency --atac-transform binary-tfidf --seed 42
python analysis/inspect_data.py --data-dir outputs/REPRO/prepared --output outputs/REPRO/data_profile.json
mkdir -p outputs/REPRO/selection_data
cp outputs/REPRO/prepared/train.csv outputs/REPRO/selection_data/train.csv
cp outputs/REPRO/prepared/validation.csv outputs/REPRO/selection_data/eval.csv
python analysis/train.py --data-dir outputs/REPRO/selection_data --output-dir outputs/REPRO/candidate_1 --method contrastive --encoder1 linear --encoder2 linear --epochs 40 --batch-size 128 --latent-dim 32 --seed 42
python analysis/train.py --data-dir outputs/REPRO/selection_data --output-dir outputs/REPRO/candidate_2 --method contrastive --encoder1 mlp --encoder2 mlp --epochs 40 --batch-size 128 --latent-dim 32 --seed 42
python analysis/train.py --data-dir outputs/REPRO/selection_data --output-dir outputs/REPRO/candidate_3 --method selfsupervised --encoder1 mlp --encoder2 linear --epochs 40 --batch-size 128 --latent-dim 32 --seed 42
for candidate in 1 2 3; do
  python analysis/evaluate.py --data-dir outputs/REPRO/selection_data --model-dir outputs/REPRO/candidate_$candidate
  python analysis/downstream.py --model-dir outputs/REPRO/candidate_$candidate
done
# Select the largest mean bidirectional validation Recall@5; for this recorded run it was candidate 1.
cp -R outputs/REPRO/candidate_1 outputs/REPRO/contrastive
python analysis/evaluate.py --data-dir outputs/REPRO/prepared --model-dir outputs/REPRO/contrastive
python analysis/downstream.py --model-dir outputs/REPRO/contrastive
```
