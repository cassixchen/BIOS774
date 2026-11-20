---
name: downstream-evaluation
description: Evaluate frozen paired multimodal embeddings on cross-modal retrieval and select models using validation Recall@5.
---

# Downstream Evaluation

## Purpose

Assess whether trained multimodal representations support paired cross-modal retrieval. Run this stage **after** `analysis/evaluate.py` has generated embeddings. Do not train or modify model weights in this stage.

## Instructions

1. Use `analysis/downstream.py` to load the embeddings produced by `analysis/evaluate.py` for the requested split. Preserve pairing and row alignment between modalities.
2. Compute retrieval in **both directions** (modality 1 → modality 2 and modality 2 → modality 1). Treat the matching row as the correct target; do not use class labels as substitutes for pair identity.
3. Report Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100, mean reciprocal rank (MRR), and median rank, for each direction and their bidirectional summary where supported. Compare with random-retrieval baselines based on the number of candidates.
4. Save downstream results in `downstream_quality_metrics.json`. Keep the combined `evaluation_metrics.json` compatible with `run_agent.py` and its report generation. Preserve `expanded_retrieval/paired_cell_ranks.csv` for per-pair ranks where applicable.
5. For **candidate selection**, use only mean bidirectional Recall@5 on the validation split. Evaluate all three candidates using the same validation split, then select the best checkpoint without retraining it.
6. Evaluate the selected checkpoint on the untouched test split **once, after selection**. Never use test metrics to choose architectures, preprocessing settings, or hyperparameters.
7. Explain downstream metrics separately from representation-quality metrics. Retrieval performance measures usefulness for paired matching; it is not, by itself, proof of clinical or biological validity.

## Workflow boundaries

- `analysis/train.py`: learns model parameters.
- `analysis/evaluate.py`: produces embeddings and `embedding_quality_metrics.json` without updating weights.
- `analysis/downstream.py`: consumes embeddings and produces retrieval metrics without updating weights.
- `run_agent.py`: orchestrates stages and model selection; Codex planning/reporting should not execute training or evaluation inside its sandbox.

Do not add KMeans, silhouette, or other clustering-based metrics unless explicitly requested.
