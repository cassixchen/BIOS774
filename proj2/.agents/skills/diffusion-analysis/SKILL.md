
# Diffusion Analysis

## Method

Use `analysis/train_diffusion.py`. Dataset modality is detected automatically.

- `image`: convolutional noise-prediction network.
- `scrna`: time-conditioned MLP on transformed expression, followed by approximate count conversion.

Choose compatible preprocessing, diffusion steps, epochs, and network size based on the objective, dataset characteristics, generation quality, and compute cost.

## Outputs

Save the model checkpoint, `training_history.csv`, `training_config.json`, and synthetic samples in the shared format.

## Evaluation

Run `analysis/evaluate_generation.py`. Assess quality and diversity, including image texture or gene-count sparsity. Short runs validate the pipeline but do not establish model quality.

