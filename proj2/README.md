# AI Agent for Generative Modeling

## Overview

This project implements an AI agent that performs generative modeling on datasets with minimal human guidance.

The system uses Codex as the reasoning and orchestration agent, together with local Python analysis tools. Given a dataset and user objective, the agent inspects the data, determines whether generative modeling is appropriate, selects preprocessing and generative methods, chooses hyperparameters, trains models, generates synthetic samples, evaluates their quality, and produces a final report.

The available generative modeling methods are:

- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)
- Normalizing Flow
- Diffusion Model

The agent selects models based on the dataset and objective rather than automatically training every available method.

## Architecture

Codex serves as the autonomous reasoning and orchestration agent, invoking local Python tools as needed.

The workflow is:

```text
Dataset + Objective
        |
        v
Data Inspection
        |
        v
Objective Interpretation
        |
        v
Preprocessing Selection
        |
        v
Model Selection
        |
        v
Model Training + Sample Generation
        |
        v
Generative Evaluation
        |
        v
Interpretation / Follow-Up
        |
        v
Final Report
```

The workflow is iterative: evaluation findings may motivate a targeted follow-up experiment or an alternative model.

Responsibilities are divided between:

- **Codex:** planning, objective interpretation, preprocessing and model selection, hyperparameter decisions, tool invocation, interpretation, follow-up experiments, and report generation.
- **Skills:** guidance for data inspection, objective interpretation, preprocessing, generative modeling methods, and evaluation.
- **Python tools:** data inspection, preprocessing, model training, synthetic sample generation, quantitative evaluation, and visualization.

`AGENTS.md` defines the overall agent workflow and constraints. Method-specific guidance is stored in `.agents/skills/`.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── AGENTS.md
├── run_agent.py
├── .agents/
│   └── skills/
│       ├── data-inspection/
│       ├── objective-interpretation/
│       ├── preprocessing/
│       ├── model-selection/
│       ├── vae-analysis/
│       ├── gan-analysis/
│       ├── flow-analysis/
│       ├── diffusion-analysis/
│       └── generative-evaluation/
├── analysis/
│   ├── inspect_data.py
│   ├── preprocessing.py
│   ├── analysis_utils.py
│   ├── train_vae.py
│   ├── train_gan.py
│   ├── train_flow.py
│   ├── train_diffusion.py
│   └── evaluate_generation.py
├── data/
│   ├── dataset_1/
│   └── dataset_2/
├── outputs/
└── report/
    └── report.pdf
```

## Evaluation Datasets

### Dataset 1: PathMNIST

Dataset 1 is a stratified subset of PathMNIST. Each observation is a 28 × 28 RGB histopathology image represented by 2,352 flattened pixel features.

The dataset contains:

- 4,500 training images
- 900 evaluation images
- 2,352 numeric pixel features
- 9 tissue classes

The `tissue_type` column contains tissue-class annotations and is excluded from unconditional generative model training.

Pixel values are stored as integers between 0 and 255. Any transformations required for model training are selected or applied by the agent and corresponding analysis scripts.

### Dataset 2: PBMC3K

Dataset 2 is derived from the PBMC3K single-cell RNA-sequencing count matrix. Each observation represents a cell, and each feature represents the raw expression count for a selected gene.

The dataset contains:

- 2,110 training cells
- 528 evaluation cells
- 2,000 gene-expression features

The `cell_type` column contains cell-type annotations and is excluded from unconditional generative model training.

Gene-expression values are stored as raw nonnegative counts. The agent determines suitable preprocessing based on the model and objective.

## Installation

The project uses Python 3.9 and requires the Codex CLI.

Install and authenticate the Codex CLI using the [official Codex instructions](https://github.com/openai/codex). Verify installation with:

```bash
codex --version
```

If needed, authenticate with:

```bash
codex login
```

Create and activate the Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

The agent uses the repository virtual environment for analysis commands.

## Single-command Execution

From the repository root, provide a dataset directory, user objective, and output directory in a single command.

**Dataset 1: PathMNIST**

```bash
.venv/bin/python3 run_agent.py \
  --data-dir data/dataset_1 \
  --objective "Generate realistic synthetic histopathology images for exploratory analysis." \
  --output-dir outputs/final_dataset_1
```

**Dataset 2: PBMC3K**

```bash
.venv/bin/python3 run_agent.py \
  --data-dir data/dataset_2 \
  --objective "Generate synthetic single-cell RNA-seq profiles for exploratory analysis." \
  --output-dir outputs/final_dataset_2
```

Each command launches Codex, which follows `AGENTS.md` and the available skills to autonomously inspect the dataset, interpret the objective, select preprocessing and generative models, choose hyperparameters, train models, generate samples, evaluate their quality, and produce a dataset-specific analysis report.

The launcher verifies that essential output artifacts exist before reporting success. Use `--dry-run` to inspect the generated instructions without executing the pipeline.

Output directories must be empty or nonexistent. Back up previous results before rerunning.

The launcher uses Codex's `workspace-write` sandbox. On Apple Silicon, this may prevent MPS GPU access and cause training to fall back to CPU. The CLI workflow does not automatically grant unsandboxed execution permissions.

The separate project-level `report/report.pdf` is prepared after both dataset experiments.

## Generated Outputs

Each dataset is analyzed independently, with generated files stored in its corresponding output directory.

```text
outputs/
├── final_dataset_1/
│   ├── data_profile.json
│   ├── objective_decision.json
│   ├── model_selection.json
│   ├── <model>/
│   │   ├── training_config.json
│   │   ├── training_history.csv
│   │   ├── model checkpoint
│   │   ├── synthetic samples
│   │   └── evaluation/
│   │       └── evaluation_metrics.json
│   └── generated_report_1.md
│
└── final_dataset_2/
    ├── data_profile.json
    ├── objective_decision.json
    ├── model_selection.json
    ├── <model>/
    │   ├── training_config.json
    │   ├── training_history.csv
    │   ├── model checkpoint
    │   ├── synthetic samples
    │   └── evaluation/
    │       └── evaluation_metrics.json
    └── generated_report_2.md
```

The exact filenames and available outputs depend on the selected models and data modality. Only models selected and executed by the agent produce model-specific artifacts.

The generated outputs include:

- **Synthetic samples:** generated images or gene-expression profiles
- **Model artifacts:** saved checkpoints, configurations, and training histories
- **Visualizations:** image grids and other visual outputs when supported
- **Quantitative evaluation:** distributional similarity, diversity, and modality-specific diagnostics
- **Decision records:** objective interpretation, model selection, and relevant follow-up decisions
- **Downstream recommendations:** assessments of appropriate uses, limitations, and risks
- **Final analysis reports:** `generated_report_1.md` and `generated_report_2.md`

## Reproducibility

Models are trained using training observations only. Learned preprocessing parameters are fitted on training data, and held-out evaluation observations are not used for model fitting.

Identifiers and labels are excluded from unconditional generative modeling. Fixed random seeds are used when supported.

Model configurations, training histories, device information, quantitative results, and relevant limitations are recorded in the generated outputs.

The analysis scripts support CUDA and Apple Metal Performance Shaders (MPS) when available, with CPU fallback. Codex's sandbox may restrict MPS access on macOS; the reported experiments were executed on CPU.

The agent autonomously determines modeling decisions, while execution permissions remain subject to user authorization. Exact model selections and results may vary across independent runs.

## Project Report

In addition to the autonomous dataset reports, `report/report.pdf` contains the manually prepared project report describing the agent architecture, autonomous decision-making process, experimental results, strengths, and limitations.
