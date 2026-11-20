---
name: self-supervised-learning
description: Guide self-supervised reconstruction-based multimodal representation learning.
---

# Self-Supervised Learning

## Instructions

Use within-modality reconstruction objectives on numeric training features when supported by `analysis/train.py`. Explain that independent reconstruction does not necessarily align paired modalities, so validation retrieval Recall@5 may be lower than contrastive alternatives. Evaluate with `analysis/evaluate.py` and `analysis/downstream.py`; do not use test data for model selection.
