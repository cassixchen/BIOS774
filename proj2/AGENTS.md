# Generative Modeling Agent

## Objective

Analyze datasets using appropriate generative models with minimal human guidance.

The agent should:
1. inspect the data and documentation;
2. interpret the objective and decide whether generation is appropriate;
3. select preprocessing, models, and hyperparameters;
4. train selected models and generate samples;
5. evaluate quality, diversity, and relevant risks;
6. recommend whether generated data should be used;
7. produce a final report.

Base decisions on the dataset, objective, model assumptions, and observed results. Do not automatically train every available model.

## Python Environment

Use the repository virtual environment:

```bash
.venv/bin/python3
```

Check dependencies before installing packages. Use MPS or CUDA when available, with CPU fallback. On Longleaf, use Slurm GPU allocations rather than login nodes.

On macOS, the Codex sandbox may restrict MPS access even when GPU acceleration is available locally. Use CPU fallback when necessary. Do not change execution permissions or bypass sandbox restrictions to enable GPU training.

Keep data inspection, evaluation, and reporting within the sandbox whenever possible. Execution permissions may require human approval, but model selection, preprocessing, hyperparameters, and evaluation decisions should remain autonomous.

## Agent Workflow

For each dataset:
1. Identify the data directory, objective, and output directory.
2. Read `DATA.md` and `schema.json` when available.
3. Run `analysis/inspect_data.py` and inspect `data_profile.json`.
4. Use `objective-interpretation` to decide whether generative modeling is justified.
5. Use `preprocessing` and `model-selection` to choose compatible methods and settings.
6. Run a short smoke test, then train selected models using the corresponding analysis scripts.
7. Evaluate generated samples using `analysis/evaluate_generation.py`.
8. Inspect results and run a small, justified follow-up experiment if necessary.
9. Write the dataset report with a recommendation and limitations.

The workflow is iterative. Do not train models when the objective does not justify generation.

When provided with a dataset directory, user objective, and output directory, execute the complete workflow without requiring the user to choose models, preprocessing, or hyperparameters. Ask for clarification only when essential information is missing.

## Available Skills

Use the relevant skills in `.agents/skills/`:

- `data-inspection`
- `objective-interpretation`
- `preprocessing`
- `model-selection`
- `vae-analysis`
- `gan-analysis`
- `flow-analysis`
- `diffusion-analysis`
- `generative-evaluation`

## Available Analysis Tools

- `analysis/inspect_data.py`
- `analysis/preprocessing.py`
- `analysis/train_vae.py`
- `analysis/train_gan.py`
- `analysis/train_flow.py`
- `analysis/train_diffusion.py`
- `analysis/evaluate_generation.py`

`analysis/analysis_utils.py` provides shared utilities.

Use supplied scripts instead of implementing new analysis code during a dataset run. If a needed capability is unavailable, report the limitation. Do not generate extra diagnostic files unless requested or produced by the tools.

## Data Inspection and Preprocessing

Choose preprocessing based on feature meaning, modality, and model requirements. Existing preprocessing strategies include `none`, `standard`, and `log_normalize`; model scripts may implement additional transformations.

Fit learned transformations on training data only. Exclude identifiers from model inputs. Use labels only when conditional modeling is explicitly justified; otherwise reserve them for interpretation.

For scRNA-seq, distinguish raw counts from normalized expression and assess sparsity, library sizes, and gene ordering. Never overwrite input data.

## Model Selection

Available families are VAE, GAN, normalizing flow, and diffusion. Select only suitable models based on data type, sample size, objective, compute, and limitations.

Use method-specific skills to choose hyperparameters. Avoid exhaustive searches; perform targeted follow-up experiments when results warrant them. Record why plausible alternatives were not selected.

## Train and Evaluation Data

Train models and fit preprocessing on training data only. Do not use the evaluation split to train models or repeatedly select hyperparameters.

Keep output directories separate across datasets, models, and follow-up runs. Use fixed seeds when supported and retain saved model configurations.

## Generative Evaluation

Use `generative-evaluation` and `analysis/evaluate_generation.py` for the models actually run.

Assess valid outputs, distributional similarity, diversity, and potential memorization. Use visual inspection for images and count/sparsity diagnostics for scRNA-seq. Test downstream utility only when the objective and available tools support it.

Do not equate low training loss with realistic samples. Do not claim biological validity, privacy guarantees, or downstream benefit without appropriate evidence. Do not directly compare metrics with different meanings.

## Final Reports

Generate `generated_report_1.md` and `generated_report_2.md` for Dataset 1 and Dataset 2, respectively, in their designated output directories. Each report should include:

- objective and relevant dataset characteristics;
- selected preprocessing and models, with reasons;
- important settings and generated outputs;
- evaluation findings and relevant baselines;
- recommendation, limitations, and warnings.

The separate project-level PDF (at most four pages, excluding references and appendices) is compiled after both dataset runs. Do not create or modify `report/report.pdf` during an individual dataset run.

Use only results actually produced. Verify files exist and reported numbers match saved outputs. Keep reporting instructions here rather than creating a reporting skill.

## Reproducibility

The project provides `run_agent.py` as its single-command entry point. It accepts a dataset directory, user objective, and output directory, then launches Codex to execute the autonomous workflow. The launcher verifies that essential output artifacts exist before reporting success.

A single invocation should allow the agent to complete the workflow autonomously using the supplied dataset and objective. The user should not need to manually select models, preprocessing, hyperparameters, or evaluation methods.

Record model configurations, seeds, devices, and meaningful warnings. Do not overwrite raw data or invent results.
