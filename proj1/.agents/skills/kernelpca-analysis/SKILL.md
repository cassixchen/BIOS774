# Kernel PCA Analysis Skill

## Purpose

Use this skill when Kernel PCA is appropriate for representing nonlinear structure that may not be captured by ordinary PCA.

Kernel PCA extends PCA by using a kernel to represent nonlinear relationships between observations.

## When to Consider Kernel PCA

## When to Consider Kernel PCA

Consider Kernel PCA when:

- nonlinear structure is plausible but a PCA-like representation is still useful;
- ordinary PCA may be too restrictive because relationships between observations are nonlinear;
- a kernel similarity can meaningfully represent the structure of the data.

Kernel PCA may be less appropriate when a linear representation is sufficient or the kernel matrix is too computationally expensive.

Compared with ordinary PCA, prefer Kernel PCA when there is a reason to model nonlinear rather than linear structure.

## Procedure

1. Review the dataset inspection and selected preprocessing strategy.

2. Select an appropriate kernel and kernel parameters.

3. Run:

   ```bash
   python3 analysis/run_kernel_pca.py \
     --data-dir <DATA_DIR> \
     --output-dir <OUTPUT_DIR> \
     --preprocessing <strategy> \
     --kernel <kernel> \
     --gamma <gamma>
   ```

4. Inspect:

   - `kernel_pca_metrics.json`
   - `kernel_pca_train.csv`
   - `kernel_pca_eval.csv`
   - `kernel_pca_train_2d.png`
   - `kernel_pca_eval_2d.png`

5. Use the Kernel PCA results together with embedding evaluation and visualization to determine whether the representation is useful.

## Hyperparameters

`kernel` determines the nonlinear similarity function used by Kernel PCA.

`gamma` controls the influence of individual observations for kernels such as the RBF kernel.

Choose the kernel and its parameters based on the data characteristics and analysis objective rather than assuming the defaults are optimal.

If the representation appears sensitive to the kernel parameters, consider a limited sensitivity analysis using another reasonable value.

## Kernel PCA Output

The implementation fits Kernel PCA using the training data and applies the fitted transformation to evaluation data.

Unlike ordinary PCA, Kernel PCA does not provide the same direct explained-variance interpretation.

Pay attention to:

- train and evaluation trustworthiness;
- structure visible in the two-dimensional embeddings;
- sensitivity to the selected kernel and its parameters when relevant.

## Interpretation

Interpret Kernel PCA as a nonlinear kernel-based representation of the data.

Do not interpret its components as explaining percentages of the original feature variance in the same way as ordinary PCA.

If labels are available, use them only for post-hoc visualization or interpretation and not for fitting Kernel PCA.

## Reporting

Report why Kernel PCA was selected, the preprocessing and hyperparameters used, and the most relevant evaluation results.

Mention important sensitivity or limitations when they affect interpretation.