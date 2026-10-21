# Model Selection

## Method

Read `data_profile.json`, `objective_decision.json`, available scripts, and compute constraints. Proceed only when generative modeling is justified.

Consider:
- **VAE:** efficient latent-variable baseline; reconstruction quality and likelihood assumptions matter.
- **GAN:** sample generation; monitor instability and mode collapse.
- **Flow:** tractable likelihood on its modeled continuous representation; assess preprocessing and invertibility constraints.
- **Diffusion:** iterative denoising; consider training and sampling cost.

Each family has an implementation for `image` and `scrna`, but compatibility does not imply suitability. For count data, assess whether the method models counts directly or converts continuous output heuristically.

## Hyperparameters

Choose architecture, learning rate, batch size, epochs, and model-specific settings based on the dataset and available compute. Start with reasonable settings and avoid exhaustive tuning.

## Evaluation

Choose relevant criteria before final evaluation: sample validity, fidelity, diversity, held-out diagnostics, and downstream utility when supported.

## Output

Save `model_selection.json` with selected models, rejected alternatives, reasons, settings, preprocessing, and evaluation plan. Do not automatically run every model.
