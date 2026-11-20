# ROCOv2 image–text representation learning

## Objective and findings

The objective was to learn representations capturing relationships between medical images and their descriptions. Exact paired image–caption retrieval in both directions is a measurable proxy for that objective. No genuine class labels were present. Of three contrastive projection architectures, the linear image and linear text encoders achieved the highest **mean bidirectional validation Recall@5: 0.1535**, and were selected using that metric alone. On the separate held-out test set, the selected checkpoint achieved mean bidirectional Recall@5 of **0.137**, compared with 0.005 for random ranking among 1,000 candidates. Retrieval and representation metrics do not establish clinical validity.

## Data, inspection, and isolation

The source was ROCOv2 in `data/dataset_1`, paired by image ID and caption. Raw inspection recorded 59,958 complete, uniquely identified official training pairs and 9,904 official validation pairs. In samples of up to 1,000 captions per split, median length was 18 words; the 10th–90th percentiles were 7–39 for training and 7–42 for official validation. No sampled training captions and one sampled official-validation caption were empty. These are sample findings, not full-corpus guarantees.

With seed 42, the preparation sampled 8,000 training and 1,000 model-selection validation pairs from the official training split, and 1,000 test pairs from the official validation split. Prepared-data checks recorded 512 image and 128 text numeric features, zero reported missing fractions, finite values, identical ordered columns, unique and pairwise-disjoint IDs, and matching manifest order with nonmissing image paths and captions. Text vocabulary, IDF and SVD were fitted on training captions only; median imputation and standardization were fitted on training features only. Validation and test were transformed using those fitted objects. The same validation split was used for all candidates; test was not used in selection. The selected validation-trained checkpoint was copied without retraining and evaluated on the held-out test. ID disjointness does not establish patient-level separation or exclude semantically similar images across splits.

## Preprocessing and model

Frozen ImageNet-pretrained ResNet18 and its associated transforms produced 512 image features. Training-fitted TF-IDF used English stop words, `min_df=2`, up to 12,000 unigram and bigram features, followed by 128-component truncated SVD. The sampled caption lengths support considering multiword terms; the vocabulary cap and SVD bound resource use. These settings were pragmatic, not benchmarked optima. A 96-dimensional color histogram plus hashing and SVD was considered cheaper but would discard image structure and lose vocabulary interpretability; unigram-only TF-IDF was also considered. None was benchmarked. Image feature extraction used MPS, with batch size 32.

All candidates learned projections over these fixed features using symmetric paired InfoNCE, normalized embeddings, temperature 0.1, Adam at learning rate 0.001, 25 epochs, batch size 128, latent dimension 64 and seed 42. Training used the actual PyTorch device **MPS**. The winning encoders are independent linear maps from 512 and 128 inputs to 64 dimensions. ResNet18 and TF-IDF/SVD were frozen, while the projection encoders were trained; this is not end-to-end CLIP or fine-tuning of the input extractors. A deeper MLP was rejected for capacity and overfitting concerns, supervised learning for lack of labels, and within-modality reconstruction because it does not directly align pairs.

## Validation model selection

Each candidate used the same 8,000 training pairs and 1,000 validation pairs; only the encoder combination differed. Scores are exact-pair Recall@5 as fractions, and the mean of the two directions is the sole selection criterion. Random Recall@5 is 0.005.

| Candidate | Image encoder | Text encoder | Image → text | Text → image | Mean Recall@5 |
|---|---|---|---:|---:|---:|
| 1, selected | Linear | Linear | 0.166 | 0.141 | **0.1535** |
| 2 | MLP | Linear | 0.120 | 0.116 | 0.1180 |
| 3 | MLP | MLP | 0.124 | 0.122 | 0.1230 |

The MLPs use one 128-unit hidden layer with ReLU. `model_selection.json` records the pre-training proposals and retains its original proposed-only status; `model_selection_results.json` records the actual winner and validation results.

## Held-out test results

Retrieval ranks the exact matching row among 1,000 candidates. Random Recall@K is K/1,000. The bidirectional mean is the arithmetic mean of the two directions.

| Metric | Image → text | Text → image | Bidirectional mean | Random recall |
|---|---:|---:|---:|---:|
| Recall@1 | 0.035 | 0.039 | 0.0370 | 0.001 |
| Recall@5 | 0.153 | 0.121 | 0.1370 | 0.005 |
| Recall@10 | 0.244 | 0.227 | 0.2355 | 0.010 |
| Recall@20 | 0.359 | 0.344 | 0.3515 | 0.020 |
| Recall@50 | 0.564 | 0.556 | 0.5600 | 0.050 |
| Recall@100 | 0.733 | 0.733 | 0.7330 | 0.100 |
| MRR | 0.1052 | 0.0997 | 0.1024 | — |
| Median rank | 37 | 40 | 38.5 | — |

The last column provides recall baselines only. The mean median rank averages the two directional medians.

For representation quality on the same test pairs, mean positive cosine was 0.3288 and mean negative cosine 0.0019, a gap of **0.3269**. The sampled positive-versus-negative pairwise AUC estimate was **0.9024**; mean squared alignment distance was **1.3424**. Image and text uniformity were **−3.7224** and **−3.6294**, and effective ranks were **45.10** and **42.90**, respectively, out of 64 dimensions. AUC and uniformity used 100,000 sampled pairs with seed 42. These summarize separation and embedding geometry, not clinical usefulness; alignment and positive cosine are related for normalized embeddings.

## Limitations and numerical checks

Similar images or captions can be false negatives during contrastive training and relevant alternatives are counted incorrect by exact-pair retrieval. ImageNet pretraining is not radiology-specific; TF-IDF/SVD can miss contextual and medical semantics. The comparison covers three architectures at one seed and fixed hyperparameters, without confidence intervals or preprocessing benchmarks. Reported test metrics describe this subset and candidate pool, and are not external validation. Prepared features were checked finite and schema-valid; the supplied artifacts do not contain a complete runtime warning log, so absence of numerical warnings during execution cannot be established from them.

## Artifacts and reproduction

`contrastive/eval_embeddings.csv` contains the saved held-out learned representations: each row has `sample_id` followed by 128 concatenated embedding values, `z0`–`z63` for image (modality 1) and `z64`–`z127` for text (modality 2). The selected weights, train-fitted projection preprocessing, training history and configuration are in `contrastive/model.pt`, `contrastive/preprocessing.joblib`, `contrastive/training_history.csv` and `contrastive/training_config.json`. Representation, retrieval and combined results remain distinct in `contrastive/embedding_quality_metrics.json`, `contrastive/downstream_quality_metrics.json` and `contrastive/evaluation_metrics.json`; per-pair ranks are in `contrastive/expanded_retrieval/paired_cell_ranks.csv`. The prepared text pipeline, manifests and features are in `prepared/`; decisions and selection results are in the JSON files in the output root.

From the repository root, the recorded inspection and preparation commands are below. Substitute a **fresh, empty** output root for `REPRO` to avoid overwriting this completed run. MPS and the pretrained weights must be available for the recorded device choice; execution environment and library versions may affect exact numerical reproducibility.

```bash
REPRO=outputs/rocov2_reproduction
mkdir -p "$REPRO"
python analysis/inspect_raw.py --source data/dataset_1 --kind rocov2 --output "$REPRO/raw_profile.json" --seed 42
python analysis/prepare_dataset_1.py --source data/dataset_1 --output "$REPRO/prepared" --train-size 8000 --validation-size 1000 --eval-size 1000 --batch-size 32 --seed 42 --device mps --image-encoder resnet18 --text-vectorizer tfidf --text-max-features 12000 --text-components 128 --text-ngram-max 2
python analysis/inspect_data.py --data-dir "$REPRO/prepared" --output "$REPRO/data_profile.json"
mkdir -p "$REPRO/model_selection_data"
cp "$REPRO/prepared/train.csv" "$REPRO/model_selection_data/train.csv"
cp "$REPRO/prepared/validation.csv" "$REPRO/model_selection_data/eval.csv"
python analysis/train.py --data-dir "$REPRO/model_selection_data" --output-dir "$REPRO/candidate_1" --method contrastive --encoder1 linear --encoder2 linear --epochs 25 --batch-size 128 --latent-dim 64 --seed 42
python analysis/train.py --data-dir "$REPRO/model_selection_data" --output-dir "$REPRO/candidate_2" --method contrastive --encoder1 mlp --encoder2 linear --epochs 25 --batch-size 128 --latent-dim 64 --seed 42
python analysis/train.py --data-dir "$REPRO/model_selection_data" --output-dir "$REPRO/candidate_3" --method contrastive --encoder1 mlp --encoder2 mlp --epochs 25 --batch-size 128 --latent-dim 64 --seed 42
for candidate in 1 2 3; do
  python analysis/evaluate.py --data-dir "$REPRO/model_selection_data" --model-dir "$REPRO/candidate_$candidate"
  python analysis/downstream.py --model-dir "$REPRO/candidate_$candidate"
done
```

Select the highest mean bidirectional validation Recall@5 before accessing test results (the recorded winner was candidate 1). For the recorded winner, the following copies only its training artifacts and evaluates the separate test split once:

```bash
mkdir -p "$REPRO/contrastive" "$REPRO/final_test_data"
cp "$REPRO/candidate_1/model.pt" "$REPRO/candidate_1/preprocessing.joblib" "$REPRO/candidate_1/training_config.json" "$REPRO/candidate_1/training_history.csv" "$REPRO/contrastive/"
cp "$REPRO/prepared/train.csv" "$REPRO/final_test_data/train.csv"
cp "$REPRO/prepared/eval.csv" "$REPRO/final_test_data/eval.csv"
python analysis/evaluate.py --data-dir "$REPRO/final_test_data" --model-dir "$REPRO/contrastive"
python analysis/downstream.py --model-dir "$REPRO/contrastive"
```

If a fresh reproduction selects a different winner, copy that candidate's training artifacts instead. These commands describe reproduction; training and evaluation were not rerun while writing this report.
