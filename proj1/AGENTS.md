# Dimension Reduction and Exploratory Data Analysis Agent

## Objective

Analyze datasets using appropriate dimension-reduction and exploratory data-analysis methods with minimal human guidance.

The agent should:

1. inspect the dataset and available documentation;
2. select appropriate preprocessing;
3. select suitable dimension-reduction methods;
4. choose appropriate hyperparameters;
5. run the selected analyses;
6. generate low-dimensional representations and visualizations;
7. evaluate and interpret the resulting embeddings using appropriate quantitative or qualitative criteria;
9. generate a final analysis report summarizing the decisions and findings.

Decisions should be based on the dataset characteristics, feature meaning, analysis objective, and observed results rather than dataset-specific assumptions.


## Python Environment

Use the repository's Python virtual environment for all analysis commands.

Run Python scripts with:

```bash
.venv/bin/python3
```

Before installing a package, verify that the repository virtual environment
is being used and that the package is not already available.

Do not install packages into temporary directories when the required
dependencies are already available in the repository environment.


## Agent Workflow

For each dataset:

1. Identify the input dataset directory and requested output directory.
2. Read `DATA.md` and `schema.json` when available.
3. Use the data-inspection skill and run `analysis/inspect_data.py`.
4. Inspect `data_profile.json`.
5. Use the preprocessing skill to select an appropriate preprocessing strategy.
6. Consider the available dimension-reduction methods and select only methods that are appropriate for the dataset and analysis objective.
7. Use the corresponding method skills and analysis scripts to run the selected methods.
8. Use the embedding-evaluation skill and run `analysis/evaluate_embeddings.py`.
9. Inspect numerical results, diagnostics, warnings, and visualizations.
10. Perform a small, targeted follow-up analysis if an important result is unstable, unclear, or sensitive to a method setting.
11. Interpret the results and generate the final analysis report.

The workflow is iterative. Results from one step may provide a reason to reconsider a previous decision.

Do not automatically run every available dimension-reduction method.


## Available Skills

Use the skills in `.agents/skills/` for statistical guidance.

Available skills include:

- `data-inspection`
- `preprocessing`
- `pca-analysis`
- `kernel-pca-analysis`
- `mds-analysis`
- `isomap-analysis`
- `lle-analysis`
- `laplacian-eigenmaps-analysis`
- `tsne-analysis`
- `umap-analysis`
- `embedding-evaluation`

Use the relevant skill to determine when an analysis is appropriate, how to choose important settings, and how to interpret the resulting output.


## Available Analysis Tools

Use the supplied analysis scripts as the primary computational tools:

- `analysis/inspect_data.py`
- `analysis/preprocessing.py`
- `analysis/run_pca.py`
- `analysis/run_kernel_pca.py`
- `analysis/run_mds.py`
- `analysis/run_isomap.py`
- `analysis/run_lle.py`
- `analysis/run_laplacian_eigenmaps.py`
- `analysis/run_tsne.py`
- `analysis/run_umap.py`
- `analysis/evaluate_embeddings.py`

`analysis/analysis_utils.py` contains shared implementation utilities used by the analysis scripts.

When an analysis is supported by an existing script, use that script rather
than reimplementing the calculation.

Use only the supplied analysis tools for computational analysis. Do not create
additional Python analysis or validation scripts during a dataset run.

The agent may perform targeted follow-up analyses by rerunning the supplied
tools with justified alternative settings. Store follow-up outputs in a
separate subdirectory of the requested output directory.

Do not generate additional diagnostic, metadata, validation, environment,
logging, or reproduction files unless they are produced by the supplied tools
or explicitly requested by the user.

If the supplied tools cannot perform an analysis that would otherwise be
useful, describe that limitation in the final report rather than implementing
a new analysis tool during the run.


## Data Inspection and Preprocessing

Inspect the dataset before selecting preprocessing or dimension-reduction methods.

Use the dataset documentation, schema, and `data_profile.json` together when making decisions.

The supported preprocessing strategies are:

- `none`
- `standard`
- `log_normalize`

Use the preprocessing skill to determine which strategy is appropriate.

Explicitly pass the selected strategy to each dimension-reduction tool:

```bash
--preprocessing <STRATEGY>
```

Do not rely on the script default during an autonomous analysis.

Do not choose preprocessing based only on the dataset name or modality.

When train and evaluation data are available, parameters estimated across observations must be fitted using training data only. Apply the corresponding transformation to the evaluation data.


## Dimension-Reduction Method Selection

Available methods are:

- PCA
- Kernel PCA
- MDS
- Isomap
- LLE
- Laplacian Eigenmaps
- t-SNE
- UMAP

Use the corresponding skill to determine whether each method is appropriate.

Select methods based on the dataset characteristics, analysis objective, method assumptions, computational requirements, and the type of structure the method is designed to preserve.

Different methods may provide complementary views of the data and do not need to be treated as competitors.

Do not run a method only because it is available.

Do not select a method solely because its visualization produces clearer separation of known labels.


## Hyperparameter Selection and Follow-Up Analysis

Start from reasonable method defaults unless the dataset characteristics or analysis objective provide a reason to use different settings.

Do not perform broad or exhaustive hyperparameter searches by default.

If an important result may depend strongly on a method setting, perform a small, targeted sensitivity analysis.

Use observations from completed analyses to decide whether follow-up analysis is necessary.

Do not tune settings solely to improve visual separation of known labels.

Record important settings used in the final analysis.


## Labels and Identifiers

Identifier columns must not be used as dimension-reduction features.

Known labels, classes, outcomes, or annotations must not be used as input features for unsupervised dimension reduction.

Labels may be used after fitting for visualization and interpretation.

Method selection, preprocessing, and hyperparameter choices should not be driven solely by which representation best separates known labels.


## Train and Evaluation Data

When separate training and evaluation datasets are available:

- fit preprocessing parameters using training data only;
- fit dimension-reduction models using training data;
- transform evaluation observations using the fitted preprocessing and
  dimension-reduction models when the method supports transformation;
- evaluate training and evaluation embeddings separately when supported.

PCA, Kernel PCA, Isomap, LLE, and UMAP support evaluation transformation in the current workflow.

MDS, Laplacian Eigenmaps, and t-SNE are training-only in the current workflow.

Do not independently fit a training-only method to the evaluation data and treat the resulting coordinates as belonging to the training embedding.


## Embedding Evaluation

After running the selected methods, use the embedding-evaluation skill and
run:

```bash
.venv/bin/python3 analysis/evaluate_embeddings.py \
  --data-dir <DATA_DIR> \
  --output-dir <OUTPUT_DIR>
```

Evaluate only methods that were actually run.

Use trustworthiness as the common quantitative measure of local-neighborhood preservation.

Also consider method-specific diagnostics produced by the analysis tools, including:

- PCA explained and cumulative variance;
- MDS stress;
- Isomap reconstruction error;
- LLE reconstruction error;
- t-SNE KL divergence.

Method-specific quantities measure different properties and should not be directly compared as if they were the same metric.

Do not automatically select the method with the highest trustworthiness.

Evaluate embeddings using quantitative results, visual inspection, the analysis objective, and the assumptions and limitations of each method.

If different preprocessing strategies were used, remember that trustworthiness is being measured relative to different preprocessed feature representations.


## Interpretation

Treat dimension-reduction results as exploratory evidence.

Use quantitative diagnostics together with visual inspection and dataset context.

Do not claim that visible groups necessarily represent true clusters, populations, states, or trajectories.

For datasets without known labels, describe visible structure without assigning unsupported identities to groups.

Distinguish local neighborhood preservation from global geometric structure.

When multiple methods are used, explain whether they reveal consistent, different, or complementary aspects of the dataset rather than forcing a single method to be declared the winner.

Inspect numerical warnings rather than automatically suppressing them. If a warning materially affects confidence in the analysis, investigate it and report the limitation.


## Final Report

Generate the final report at the location requested by the user.

The report must be based only on analyses and outputs actually produced during the current run.

Include the information needed to understand the analysis, such as:

- important dataset characteristics;
- selected preprocessing and the reason for the choice;
- selected dimension-reduction methods and why they were appropriate;
- selected appropriate hyperparameters;
- quantitative evaluation results;
- relevant visual findings;
- targeted follow-up analyses, when performed;
- interpretation of the embeddings;
- important limitations or warnings.

Briefly explain why an available method was not selected.

Do not invent results for analyses that were not performed.

Ensure that numerical values in the report match the generated output files.

Use labels only for post-hoc interpretation when labels are available.


## Reproducibility and Output Management

Store generated files in the output directory specified for the analysis.

Do not modify or overwrite input data.

Use fixed random seeds when supported unless there is a specific analytical reason not to do so.

Keep outputs from different datasets or analysis runs in separate directories.

Generated outputs should make it possible to determine:

- what preprocessing was used;
- which methods were run;
- what important settings were used;
- what quantitative results were obtained;
- what warnings or limitations occurred.

Before completing the analysis, verify that the requested final report and relevant output files exist.