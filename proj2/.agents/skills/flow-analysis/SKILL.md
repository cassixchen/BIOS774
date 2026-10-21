
# Flow Analysis

## Method

Use `analysis/train_flow.py`. Dataset modality is detected automatically.

The script applies selected preprocessing, fits PCA on training data, standardizes PCA coordinates, and trains a RealNVP flow. Synthetic samples are reconstructed into image pixels or approximate integer gene counts.

Choose compatible preprocessing, PCA dimension, coupling layers, and training settings based on the dataset, objective, and compute budget.

## Outputs

Save `flow_model.pt`, `training_history.csv`, `training_config.json`, and synthetic samples in the shared format.

## Evaluation

Run `analysis/evaluate_generation.py`. Report PCA explained variance and sample-quality diagnostics. Flow likelihood applies to PCA coordinates, not directly to original pixels or counts; count conversion is heuristic.

