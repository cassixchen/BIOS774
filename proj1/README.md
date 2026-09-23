# AI Agent for Dimension Reduction and Exploratory Data Analysis

## Overview

This project implements an AI agent that performs exploratory dimension-reduction analysis on unseen datasets with minimal human guidance.

The system uses Codex as the reasoning and orchestration agent together with deterministic local Python tools. The agent inspects a dataset, selects appropriate preprocessing and dimension-reduction methods, chooses method settings, executes the analysis, evaluates the resulting embeddings, and generates a final report.

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

The system is intended to be executed through Codex, which acts as the autonomous reasoning agent and invokes the repository's local Python analysis tools as needed.

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
│   ├── dataset_1/
│   └── dataset_2/
├── outputs/
└── report/
    └── report.pdf
```

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

`tissue_type` contains the PathMNIST tissue-class name and is retained for post-hoc visualization and qualitative interpretation but is excluded from preprocessing, dimension-reduction fitting, hyperparameter selection, and quantitative embedding evaluation.

No scaling or other feature preprocessing is applied during dataset construction.

### Dataset 2: PBMC3K

Dataset 2 is derived from the PBMC3K single-cell RNA-sequencing count matrix. Each observation represents a cell, and each analysis feature represents the raw expression count for a selected gene.

The original matrix contains 2,700 cells and 32,738 genes. Cell-type annotations from the processed Seurat PBMC3K dataset are matched to the original cells using their cell barcodes. Of the 2,700 original cells, 2,638 have corresponding cell-type annotations and are retained for analysis. The remaining 62 cells are excluded.

The 2,638 retained cells are split into training and evaluation sets using an 80/20 split with random seed 123.

Genes detected in at least three training cells are retained as candidates. The 2,000 genes with the highest raw-count variance in the training data are then selected and used for both the training and evaluation data.

The resulting dataset contains:

- 2,110 training cells;
- 528 evaluation cells;
- 2,000 gene-expression features;
- 9 cell types.

`cell_type` contains the Seurat cell-type annotation and is retained for post-hoc visualization and qualitative interpretation but is excluded from preprocessing, dimension-reduction fitting, hyperparameter selection, and quantitative embedding evaluation.

The feature values remain raw nonnegative counts during dataset construction. No library-size normalization, log transformation, standardization, or other analysis preprocessing is applied during dataset construction.


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

he generated `train.csv` and `eval.csv` files are not included in the repository and must first be created using the dataset construction scripts.

From the repository root, run:

```bash
python3 data/dataset_1/create_dataset_1.py
python3 data/dataset_2/create_dataset_2.py
```

These scripts create the training and evaluation datasets used by the agent:

```text
data/dataset_1/train.csv
data/dataset_1/eval.csv
data/dataset_2/train.csv
data/dataset_2/eval.csv
```

After constructing the datasets, run **Codex from the repository root** and ask it to analyze the desired dataset (for example, `data/dataset_1` or `data/dataset_2`).

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

In addition to the autonomous dataset reports, `report/report.pdf` contains the manually prepared project report describing the agent architecture, decision-making process, experimental results, strengths, and limitations.