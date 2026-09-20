# AI Agent for Dimension Reduction and Exploratory Data Analysis

## Overview

This project implements an AI agent that performs exploratory dimension-reduction analysis on unseen datasets with minimal human guidance.

The system uses **Codex as the reasoning and orchestration agent** together with deterministic local Python tools. The agent inspects a dataset, selects appropriate preprocessing and dimension-reduction methods, chooses method settings, executes the analysis, evaluates the resulting embeddings, and generates a final report.

The available dimension-reduction methods are:

- PCA
- Kernel PCA
- MDS
- Isomap
- LLE
- Laplacian Eigenmaps
- t-SNE
- UMAP

The agent selects methods based on the dataset and analysis objective rather than automatically running every available method.


## Architecture

The analysis follows an iterative workflow:

```text
Dataset + Objective
        ↓
Data Inspection
        ↓
Preprocessing Selection
        ↓
Method Selection
        ↓
Dimension Reduction
        ↓
Embedding Evaluation
        ↓
Interpretation / Follow-Up
        ↓
Final Report
```

Responsibilities are divided between:

- **Codex:** planning, preprocessing and method selection, hyperparameter decisions, tool invocation, interpretation, follow-up analysis, and report generation.
- **Skills:** statistical guidance for inspection, preprocessing, dimension-reduction methods, and evaluation.
- **Python tools:** deterministic data inspection, preprocessing, dimension reduction, metric calculation, and visualization.

`AGENTS.md` defines the overall agent workflow and constraints. Method-specific guidance is stored in `.agents/skills/`.


## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── AGENTS.md
├── .agents/
│   └── skills/
│       ├── data-inspection/
│       ├── preprocessing/
│       ├── pca-analysis/
│       ├── kernel-pca-analysis/
│       ├── mds-analysis/
│       ├── isomap-analysis/
│       ├── lle-analysis/
│       ├── laplacian-eigenmaps-analysis/
│       ├── tsne-analysis/
│       ├── umap-analysis/
│       └── embedding-evaluation/
├── analysis/
│   ├── inspect_data.py
│   ├── preprocessing.py
│   ├── analysis_utils.py
│   ├── run_pca.py
│   ├── run_kernel_pca.py
│   ├── run_mds.py
│   ├── run_isomap.py
│   ├── run_lle.py
│   ├── run_laplacian_eigenmaps.py
│   ├── run_tsne.py
│   ├── run_umap.py
│   └── evaluate_embeddings.py
├── data/
│   ├── dataset_0/
│   ├── dataset_1/
│   └── dataset_2/
├── outputs/
└── report/
    └── report.tex
```

`dataset_0` is used for development and workflow testing.

The two final evaluation datasets are:

- **Dataset 1:** PathMNIST image data
- **Dataset 2:** PBMC3K single-cell RNA-sequencing data


## Evaluation Datasets

### Dataset 1: PathMNIST

Dataset 1 is a stratified subset of PathMNIST. Each observation is a 28 × 28 RGB image represented by 2,352 flattened pixel features.

The dataset contains:

- 4,500 training images;
- 900 evaluation images;
- 2,352 numeric pixel features;
- 9 tissue classes.

The training subset contains 500 observations from each class, and the evaluation subset contains 100 observations from each class.

`tissue_type` is retained for post-hoc visualization and interpretation but is excluded from dimension-reduction fitting.

### Dataset 2: PBMC3K

Dataset 2 is derived from the PBMC3K single-cell RNA-sequencing count matrix.

The original matrix contains 2,700 cells and 32,738 genes. Cells are split into 2,160 training cells and 540 evaluation cells before gene filtering and selection.

Genes detected in at least three training cells are retained as candidates. The 2,000 genes with the highest raw-count variance in the training data are then selected and used for both training and evaluation data.

The resulting dataset contains:

- 2,160 training cells;
- 540 evaluation cells;
- 2,000 gene-expression features.

The feature values remain raw counts during dataset construction. No cell-type labels are supplied.


## Installation

The project uses Python 3.9.

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

The agent uses the repository virtual environment for analysis commands.


## Running the Agent

The user provides a dataset directory, analysis objective, output directory, and requested report location. The agent determines the preprocessing, dimension-reduction methods, method settings, evaluation, and any necessary follow-up analysis.

Example for Dataset 1:

```text
Analyze the dataset in data/dataset_1 using the repository instructions and available skills. Store all analysis outputs in outputs/final_dataset_1. Perform the dimension-reduction and exploratory data analysis autonomously with minimal manual guidance. Use the dataset documentation and schema when making analysis decisions. Create the final analysis report as outputs/final_dataset_1/generated_report_1.md.
```

Example for Dataset 2:

```text
Analyze the dataset in data/dataset_2 using the repository instructions and available skills. Store all analysis outputs in outputs/final_dataset_2. Perform the dimension-reduction and exploratory data analysis autonomously with minimal manual guidance. Use the dataset documentation and schema when making analysis decisions. Create the final analysis report as outputs/final_dataset_2/generated_report_2.md.
```


## Generated Outputs

Each dataset is analyzed independently, with generated files stored in its corresponding output directory.

```text
outputs/
├── final_dataset_1/
│   ├── data_profile.json
│   ├── <method>_train.csv
│   ├── <method>_eval.csv
│   ├── <method>_train_2d.png
│   ├── <method>_eval_2d.png
│   ├── <method>_metrics.json
│   ├── embedding_evaluation.json
│   └── generated_report_1.md
│
└── final_dataset_2/
    ├── data_profile.json
    ├── <method>_train.csv
    ├── <method>_eval.csv
    ├── <method>_train_2d.png
    ├── <method>_eval_2d.png
    ├── <method>_metrics.json
    ├── embedding_evaluation.json
    └── generated_report_2.md
```

Only files for methods selected and run by the agent are generated. Methods without an out-of-sample transform produce training outputs only.

The generated outputs provide:

- **Visualizations:** two-dimensional embedding plots (`*_2d.png`)
- **Quantitative evaluation:** method-specific metrics (`*_metrics.json`) and common embedding evaluation (`embedding_evaluation.json`)
- **Selected methods and preprocessing:** recorded in the method outputs and summarized in the generated report
- **Final analysis reports:** `generated_report_1.md` and `generated_report_2.md`


## Reproducibility

The workflow is designed to avoid data leakage and support reproducible analysis.

Preprocessing parameters are fitted using training data when necessary, and held-out observations reuse the fitted transformations. Dimension-reduction models are fitted on training observations when out-of-sample transformation is supported.

Identifiers and labels are excluded from unsupervised fitting, and fixed random seeds are used for stochastic methods when supported.

Method settings, preprocessing choices, quantitative results, and relevant limitations are recorded in the generated outputs.


## Project Report

In addition to the autonomous dataset reports, `report/report.tex` contains the manually prepared project report describing the agent architecture, decision-making process, experimental results, strengths, and limitations.