---
name: data-inspection
description: Inspect raw and prepared multimodal datasets before model selection and training.
---

# Data Inspection

## Instructions

Inspect raw inputs with `analysis/inspect_raw.py` before preprocessing, and inspect numeric prepared inputs with `analysis/inspect_data.py` afterward. Raw data are `data/dataset_1` (ROCOv2 image-caption pairs) and `data/dataset_2` (paired Multiome RNA/ATAC). Save raw inspection as `raw_profile.json`, prepared inspection as `data_profile.json`. Check sample pairing, modality dimensions, missingness, feature ranges, split sizes, and alignment. Keep validation/test separate from training and never use their statistics to fit transforms.
