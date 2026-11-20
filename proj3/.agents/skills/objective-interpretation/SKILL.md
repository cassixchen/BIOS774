---
name: objective-interpretation
description: Translate broad representation learning objectives into measurable dataset-specific goals.
---

# Objective Interpretation

## Instructions

Translate the user's broad multimodal representation-learning objective into dataset-specific, measurable goals without narrowing it unnecessarily. For dataset_1, learn image-text representations; for dataset_2, learn paired RNA-ATAC representations. Note whether labels are available and whether pair alignment is meaningful. Three candidates are proposed after data inspection; host `analysis/train.py`, `analysis/evaluate.py`, and `analysis/downstream.py` execute training and two distinct evaluation stages. Validation mean bidirectional Recall@5 is the selection criterion; test remains untouched until the winner is selected.
