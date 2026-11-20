---
name: contrastive-learning
description: Guide contrastive multimodal representation learning with paired InfoNCE objectives and consistent evaluation.
---

# Contrastive Learning

## Instructions

Use paired symmetric InfoNCE across two learned encoders, with aligned rows as positives and other batch rows as negatives. Explain temperature, possible false negatives, and why retrieval-based selection can favor contrastive training. Training is performed by host `analysis/train.py`; representation quality is assessed by `analysis/evaluate.py`, and paired retrieval by `analysis/downstream.py`. Never confuse frozen raw feature extractors with trained projection encoders.
