
# GAN Analysis

## Method

Use `analysis/train_gan.py`. Dataset modality is detected automatically.

- `image`: convolutional GAN.
- `scrna`: fully connected GAN on transformed expression, with approximate conversion to counts.

Choose compatible preprocessing and training settings based on dataset characteristics, objective, and stability. Pass the selected method using `--preprocessing`.

Monitor generator and discriminator losses, but do not use them as the sole quality measure.

## Outputs

Save `gan_model.pt`, `training_history.csv`, `training_config.json`, and synthetic samples in the shared format.

## Evaluation

Run `analysis/evaluate_generation.py`. Check diversity, mode collapse, image realism, and scRNA-seq sparsity and gene-detection patterns as appropriate.

