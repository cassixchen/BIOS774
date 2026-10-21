
---
name: vae-analysis
description: Train a VAE for image or scRNA-seq generation when selected.
---

# VAE Analysis

## Method

Use `analysis/train_vae.py`. Dataset modality is detected automatically.

- `image`: convolutional VAE with Gaussian latent variables.
- `scrna`: count-aware VAE with a negative-binomial reconstruction model.

Choose preprocessing, latent dimension, KL weight, and training settings based on the data and objective. Pass the selected preprocessing using `--preprocessing`.

## Outputs

Save `vae_model.pt`, `training_history.csv`, `training_config.json`, and synthetic samples (`synthetic_images.npy` or `synthetic_counts.csv`). Image runs also save `synthetic_grid.png`.

## Evaluation

Run `analysis/evaluate_generation.py`. Assess fidelity and diversity, not training loss alone. Preserve the training/evaluation split and report limitations.

