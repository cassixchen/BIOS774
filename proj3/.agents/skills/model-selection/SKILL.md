---
name: model-selection
description: Propose and select multimodal model candidates using validation retrieval metrics only.
---

# Model Selection

## Instructions

Inspect the prepared dataset and user objective, then propose exactly three feasible candidate configurations, each specifying method (`contrastive`, `selfsupervised`, or `supervised` only if labels exist), independent `encoder1` and `encoder2` (`linear`, `mlp`, `deep-mlp`), epochs, batch size, latent dimension, and seed. Explain rationale and alternatives in `model_selection.json`. The host launcher runs `analysis/train.py`, then `analysis/evaluate.py`, then `analysis/downstream.py` for each candidate. Select using ONLY mean bidirectional validation Recall@5 from downstream evaluation, comparing candidates on the same validation split. Copy the selected validation-trained checkpoint; do not retrain on test or select using test results.
