---
name: representation-evaluation
description: Evaluate learned multimodal embedding quality and export embeddings for downstream tasks.
---

# Representation Evaluation

## Purpose

Measure the structure and cross-modal alignment of learned representations without changing model weights.

## Instructions

1. After `analysis/train.py`, run `analysis/evaluate.py` on the specified validation or test split using the trained checkpoint.
2. Generate paired embeddings for both modalities, preserving row order and sample identifiers for downstream retrieval.
3. Compute positive and negative paired cosine similarity, similarity gap, pairwise AUC, alignment, within-modality uniformity, and effective rank where supported.
4. Save representation-quality metrics to `embedding_quality_metrics.json` and export embeddings for reuse by `analysis/downstream.py`. Produce relevant PCA visualizations where supported.
5. Preserve compatibility with the combined `evaluation_metrics.json` consumed by `run_agent.py` and the final report.
6. Interpret metrics carefully: for normalized embeddings, squared alignment distance is related to positive cosine similarity; embedding metrics do not establish biological or clinical validity.

## Boundaries

- `analysis/evaluate.py` measures **representation quality**, not retrieval rankings.
- `analysis/downstream.py` handles **downstream retrieval**, including Recall@K, MRR, median rank, and random baselines. Follow the `downstream-evaluation` skill for those instructions.
- Never retrain or update weights during evaluation, and never use the untouched test set for model selection.
- Codex planning/reporting should not run training or evaluation inside its sandbox; `run_agent.py` orchestrates execution outside it.

Do not add KMeans or silhouette metrics unless explicitly requested.
