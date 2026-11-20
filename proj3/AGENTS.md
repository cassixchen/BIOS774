# Autonomous Multimodal Representation-Learning Agent

## Mission
Given a raw or already prepared paired multimodal dataset and a user objective, autonomously inspect the data, choose feasible preprocessing and representation-learning methods, execute the workflow, evaluate held-out results, and produce a reproducible report. Make decisions based on observed dataset properties, not just script defaults. Never fabricate metrics or claim unsupported capabilities.

## Safety and reproducibility
- Do not modify raw source data, existing model artifacts, completed runs, or previously prepared datasets.
- Use the output paths supplied by `run_agent.py`. If an output path is already populated, stop and request a fresh path rather than overwriting it.
- Use seed 42 unless there is a documented reason to choose otherwise. Record all commands, seeds, parameters, and selected/rejected alternatives.
- Keep training and evaluation data disjoint. Fit feature ranking, vocabulary, IDF, SVD, imputers, and scalers on training data only. Never fit transformations on evaluation data or use evaluation results to select preprocessing.
- Never use sample identifiers or labels as model input features. Do not invent labels, especially cell-type labels for Multiome.
- Do not conceal numerical warnings. Check for nonfinite values and explain any unresolved warnings in the report.
- Read relevant `.agents/skills/*/SKILL.md` instructions for inspection, preprocessing, objective interpretation, model selection, training, and evaluation.

## Step 1 — Inspect the raw data FIRST
If the source is raw, run `analysis/inspect_raw.py` with the appropriate `--kind` (`rocov2` or `multiome`) and save the JSON output as `raw_profile.json` in the requested output directory. Read the complete profile before choosing or running any preparation command.

Review the supported preprocessing options using `analysis/prepare_dataset_1.py --help` or `analysis/prepare_dataset_2.py --help`. Do not assume default values are optimal. If the input is already a prepared numeric dataset, skip raw preparation and begin with `analysis/inspect_data.py`.

## Step 2 — Autonomously select preprocessing
Before executing preprocessing, write `preprocessing_decision.json` in the requested output directory. Include:
1. Concrete inspection measurements used to make decisions.
2. At least two feasible preprocessing configurations, with advantages, disadvantages, and compute costs.
3. The selected methods and parameters, including exact preparation command.
4. A specific, evidence-based justification for each modality's normalization/transformation, feature-selection method, feature count, and dimensionality reduction.
5. Rejected alternatives and reasons for rejection.
6. How the workflow prevents train/evaluation leakage and handles resource limitations.

Do not simply repeat the defaults. A method may coincide with a default if the inspection supports it, but explain why. Do not claim to have benchmarked alternatives unless they were actually run and evaluated with a valid training-only selection procedure.

### ROCOv2: supported alternatives
- Image representation: `--image-encoder resnet18|color-histogram`. ResNet18 uses frozen pretrained weights and provides semantic-oriented features; a color histogram is a cheap, nonsemantic baseline. Consider the retrieval objective, compute resources, and pretrained-weight availability.
- Caption representation: `--text-vectorizer tfidf|hashing`. Consider vocabulary interpretability, corpus size, and memory.
- Text configuration: `--text-max-features`, `--text-components`, `--text-ngram-max 1|2` and other options shown by `--help`. Consider sampled caption lengths, corpus size, and compute.
- Sample sizes, batch size, seed, and device are configurable.
- This workflow uses fixed image/text feature extraction followed by learned projections; do NOT call it end-to-end CLIP or claim that ResNet18 or the text vectorizer was fine-tuned.

### 10x Multiome: supported alternatives
- RNA normalization: `--rna-normalization library-log1p|log1p`. Consider observed RNA library-size variation.
- RNA feature selection: `--rna-feature-selection variance|detection`. Consider gene detection and variance patterns.
- ATAC transformation: `--atac-transform tfidf|binary-tfidf`. Consider count magnitudes and whether accessibility presence is the intended signal.
- ATAC feature selection: `--atac-feature-selection frequency|variance`.
- Select `--rna-features`, `--atac-features`, `--rna-components`, and `--atac-components` based on dimensions, sparsity, sample size, and feasibility.
- Preserve paired cell barcodes. Do not densify the complete high-dimensional matrix. Do not invent cell-type labels.

## Step 3 — Prepare and validate
Run the selected `analysis/prepare_dataset_1.py` or `analysis/prepare_dataset_2.py` command once, writing to the fresh prepared-data path supplied by the agent launcher. Confirm that `train.csv` and `eval.csv` exist.

Run `analysis/inspect_data.py` and save `data_profile.json` in the requested output directory. Validate both modalities, numerical feature types, finite values, consistent train/eval columns, unique sample IDs, and pairing/alignment. Stop on invalid or mismatched data; do not silently proceed.

## Step 4 — Interpret the objective and choose a model
Write `objective_decision.json` explaining whether the task is paired retrieval, classification, clustering, or exploratory representation learning, and whether genuine labels are present.

Write `model_selection.json` recording the selected method, **encoder1 and encoder2**, architecture, hyperparameters, rationale, and rejected alternatives. Use `analysis/train.py --help` to inspect supported methods:
- Prefer **contrastive** (CLIP-style symmetric InfoNCE over fixed input features) for aligned image/text or RNA/ATAC pair retrieval.
- Use **supervised** learning only when genuine, suitable training labels are available and the objective calls for it.
- Consider **selfsupervised** for reconstruction or label-free exploration when pairwise alignment is not the primary objective. This method is specifically a within-modality **autoencoder** with mean-squared-error reconstruction, NOT masked reconstruction or cross-modal prediction.

After raw inspection and preprocessing, use `data_profile.json` and the chosen preprocessing settings to select the **encoder for each modality separately**: `--encoder1 linear|mlp|deep-mlp` and `--encoder2 linear|mlp|deep-mlp`. Consider actual feature dimensions, sample count, preprocessing representation, objective, capacity and overfitting risk. Compare at least two feasible encoder combinations conceptually, record selected/rejected architectures and reasoning in `model_selection.json`, and pass both flags to the training command. Do not always choose MLP merely because it is the default. These are projection encoders over precomputed features, not pretrained foundation-model fine-tuning.

Choose and document epochs, batch size, latent dimension, and seed before looking at held-out evaluation results. Use the available `.agents/skills` guidance.

## Step 5 — Train and evaluate
The host `run_agent.py` executes training and evaluation outside the Codex sandbox after Codex writes a validated `execution_plan.json`. During the planning Codex phase, DO NOT execute training/evaluation; during the reporting Codex phase, DO NOT rerun them. The execution plan must contain exactly `method`, `encoder1`, `encoder2`, `epochs`, `batch_size`, `latent_dim`, `seed`. The host records the actual PyTorch device in `training_config.json`.

Train with `analysis/train.py`, saving artifacts in a method-specific subdirectory of the requested output directory (for example `contrastive/`). Evaluate with `analysis/evaluate.py` using the untouched held-out split.

For paired retrieval, report both directions of Recall@1, @5, and @10 with random-ranking baselines. Save embeddings, model weights, preprocessing objects, training history, metrics, and plots. Exploratory KMeans clusters and PCA are descriptive only, not proof of meaningful medical or biological groups. Report label-based probes only if actual labels exist.

## Step 6 — Generate the report and verify artifacts
Create the requested `generated_report_1.md` or `generated_report_2.md` in the requested output directory. Include:
- Objective, dataset provenance, raw inspection findings, sample counts, and split policy.
- Selected and rejected preprocessing methods, linked to measured data characteristics.
- Model architecture, learning objective, hyperparameters, training and evaluation protocol.
- Verified numerical results and a clear comparison to appropriate baselines.
- Limitations, numerical warnings, possible false negatives, and whether features are frozen.
- Exact reproduction commands and paths to saved artifacts.

Verify that all required artifacts exist and are nonempty before announcing completion. Do not create or overwrite `report/report.pdf`; that is a separate final deliverable.


## Split evaluation scripts (current)
The host runs `analysis/train.py`, `analysis/evaluate.py` (representation quality), and `analysis/downstream.py` (retrieval and validation Recall@5) in order. Do not run these inside Codex planning/reporting. Keep `embedding_quality_metrics.json`, `downstream_quality_metrics.json`, and `evaluation_metrics.json` distinct. Prepared features belong inside each run output directory.
