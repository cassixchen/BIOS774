---
name: preprocessing
description: Prepare ROCOv2 and Multiome datasets with training-only fitted transformations.
---

# Preprocessing

## Instructions

Use `analysis/prepare_dataset_1.py` for ROCOv2 and `analysis/prepare_dataset_2.py` for Multiome. Inspect the selected script's `--help` and record preprocessing rationale in `preprocessing_decision.json` before preparing. Write prepared artifacts inside the requested run's `outputs/final_dataset_1/prepared/` or `outputs/final_dataset_2/prepared/`, not under `data/`. Produce `train.csv`, `validation.csv`, and `eval.csv`. Dataset 1 targets 8000/1000/1000 pairs; dataset 2 targets 8776/1097/1097 cells when 10970 are present. Fit text transforms, feature selection, IDF, SVD, and scaling on training samples only; apply frozen transforms to validation and test. Never train on evaluation labels or features.
