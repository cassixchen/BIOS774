---
name: supervised-learning
description: Guide supervised multimodal representation learning when genuine labels are available.
---

# Supervised Learning

## Instructions

Use supervised learning only when genuine training labels are available and supported by `analysis/train.py`. Do not invent labels from row IDs, captions, or evaluation pairs. Report class balance and limitations. Evaluate learned representations with `analysis/evaluate.py` and downstream paired retrieval with `analysis/downstream.py`, using the same validation-only selection protocol.
