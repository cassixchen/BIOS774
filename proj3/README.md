# AI Agent for Multimodal Representation Learning

## Overview

This project develops an AI agent for multimodal representation learning on biomedical datasets. Codex CLI interprets a user-defined objective and inspects the data to select preprocessing, learning methods, and encoder configurations. Python tools perform feature extraction, training, evaluation, and reporting.

The agent compares candidate models using **mean bidirectional validation Recall@5**, selects the best candidate, and evaluates its saved checkpoint on held-out test data. The experiments emphasize reproducible, data-dependent agent decisions rather than maximizing benchmark performance.

## Workflow

```text
Dataset + User Objective
          |
          v
  Raw Data Inspection
          |
          v
 Objective Interpretation
          |
          v
 Preprocessing Selection
          |
          v
 Prepared Data Inspection
          |
          v
 Candidate Method / Encoder Planning
          |
          v
   Candidate Training
          |
          v
 Validation Retrieval Evaluation
          |
          v
 Best Candidate Selection
          |
          v
 Held-out Test Evaluation
          |
          v
 Embeddings + Metrics + Final Report
```

Codex CLI provides the reasoning, guided by `AGENTS.md` and task-specific instructions in `.agents/skills/`. The `run_agent.py` launcher coordinates deterministic Python tools and separates validation-based model selection from final test evaluation.

## Main Components

| Component | Role |
|---|---|
| `run_agent.py` | Launches and coordinates the autonomous workflow |
| `AGENTS.md` | Repository-wide instructions for the agent |
| `.agents/skills/` | Task-specific agent guidance |
| `analysis/inspect_raw.py` | Examines raw modalities, sample counts, and data quality |
| `analysis/prepare_dataset_1.py` | Prepares ROCOv2 image and text features |
| `analysis/prepare_dataset_2.py` | Prepares Multiome RNA and ATAC features |
| `analysis/inspect_data.py` | Checks prepared features and split/pair alignment |
| `analysis/train.py` | Trains candidate representation models |
| `analysis/evaluate.py` | Computes embedding-quality diagnostics |
| `analysis/downstream.py` | Computes bidirectional cross-modal retrieval metrics |

## Datasets

### Dataset 1: ROCOv2 — Radiology Images and Text

[ROCOv2](https://doi.org/10.5281/zenodo.8333645) provides paired radiology images and descriptions. The experiment uses 8,000 training pairs and 1,000 validation pairs sampled from the official training split, plus 1,000 held-out test pairs sampled from the official validation split.

- **Image features:** Frozen ImageNet-pretrained ResNet18, producing 512-dimensional vectors.
- **Text features:** Training-fitted TF-IDF with unigrams and bigrams, followed by truncated SVD to 128 dimensions.
- **Learning method:** Contrastive learning using symmetric InfoNCE.
- **Embeddings:** 64 dimensions per modality.
- **Training:** Batch size 128; 25 epochs.

The agent compared three contrastive encoder configurations: Linear / Linear, MLP / Linear, and MLP / MLP. The **Linear / Linear** configuration achieved the highest validation mean bidirectional Recall@5 (15.35%). Medical concept annotations available in the source dataset were not used as supervised labels in this experiment.

### Dataset 2: 10x Multiome — Paired RNA and ATAC

The [10x Genomics Multiome dataset](https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0) contains paired gene-expression and chromatin-accessibility measurements for 10,970 cells. The experiment uses 8,776 training, 1,097 validation, and 1,097 held-out test cells.

- **RNA preprocessing:** Library-size normalization to 10,000 counts per cell, log transformation, and selection of 2,000 highly variable genes from training data.
- **ATAC preprocessing:** Binarization, TF-IDF, and selection of 12,000 peaks from training data.
- **Dimensionality reduction:** Training-fitted truncated SVD to 64 features per modality.
- **Embeddings:** 32 dimensions per modality.
- **Training:** Batch size 256; 30 epochs.

The agent compared contrastive encoder configurations with a self-supervised reconstruction candidate. Contrastive candidates minimized symmetric InfoNCE; the reconstruction candidate used a reconstruction loss. **Linear / Linear contrastive learning** achieved the highest validation mean bidirectional Recall@5 (8.204%).

## Installation

Install Python dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Install and authenticate [Codex CLI](https://github.com/openai/codex) following its official instructions:

```bash
codex --version
codex login
```

Place the datasets in their expected local directories (for example, `data/dataset_1/` and `data/dataset_2/`). Large raw datasets and generated model artifacts need not be committed to Git.

## Running the Agent

The workflow starts from `run_agent.py`. The following commands illustrate the intended dataset/objective/output interface; **check `python run_agent.py --help` for the exact flags supported by your local version**.

**ROCOv2:**

```bash
python run_agent.py \
  --data-dir data/dataset_1 \
  --objective "Learn aligned image and text representations for cross-modal retrieval." \
  --output-dir outputs/final_dataset_1
```

**10x Multiome:**

```bash
python run_agent.py \
  --data-dir data/dataset_2 \
  --objective "Learn aligned RNA and ATAC representations from paired single-cell measurements." \
  --output-dir outputs/final_dataset_2
```

## Evaluation

Candidate selection uses **mean bidirectional Recall@5 on the validation set**. After selection, the existing winning checkpoint is evaluated on held-out test pairs without retraining or test-based model selection.

Representation quality is assessed using matched and unmatched cosine similarity, cosine similarity gap, pairwise AUC, alignment, uniformity, and effective rank. Downstream evaluation uses exact-pair bidirectional retrieval, including Recall@K, mean reciprocal rank (MRR), median rank, and random-ranking baselines.

### Results

| Metric | ROCOv2 | 10x Multiome |
|---|---:|---:|
| Selected method | Contrastive | Contrastive |
| Selected encoders | Linear / Linear | Linear / Linear |
| Validation mean R@5 | 15.35% | 8.204% |
| Held-out test mean R@5 | 13.70% | 9.03% |
| Held-out test pairwise AUC | 0.9024 | 0.9015 |
| Held-out test mean R@100 | 73.30% | 63.04% |

Both selected models outperform random-ranking baselines in exact-pair retrieval. These metrics do **not** establish clinical utility, biological validity, or performance on unrelated downstream tasks.

## Outputs and Reproducibility

The agent produces decision records, dataset profiles, preprocessing artifacts, model configurations and checkpoints, training histories, learned embeddings, evaluation metrics, retrieval results, and dataset-specific generated reports.

Preprocessing transformations are fitted on training observations only. All candidates within a dataset use the same training/validation partition. Validation metrics determine the selected candidate; held-out test metrics are reported only after selection. The recorded experiments used PyTorch with Apple Metal Performance Shaders (MPS) acceleration.

## References

- [ROCOv2 dataset — Zenodo](https://doi.org/10.5281/zenodo.8333645)
- [10x Genomics 10k PBMC Multiome dataset](https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0)
- [OpenAI Codex CLI](https://github.com/openai/codex)
