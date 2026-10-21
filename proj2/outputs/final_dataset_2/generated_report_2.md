# Dataset 2: synthetic single-cell RNA-seq analysis

## Objective and decision

Generate synthetic single-cell RNA-seq profiles for exploratory analysis. Decision: **GENERATIVE_APPROPRIATE**. A count-aware generative baseline is justified, but producing valid counts alone does not establish biological realism. Final recommendation: **IMPROVE_MODEL**; these samples are suitable only as clearly marked synthetic inputs for software/workflow demonstrations, pending stronger validation for biological exploration.

## Data inspection

The supplied PBMC3K subset contains 2,110 training cells, 528 held-out cells, and 2,000 genes. Documentation states that 2,638 annotated cells were retained from the original 2,700 cells; the split used seed 123. Genes were selected by raw-count variance using training cells only, after requiring detection in at least three training cells. This is a selected-gene matrix, not a full transcriptome.

`analysis/inspect_data.py` produced [data_profile.json](data_profile.json). Training counts have no missing entries and 74.0048% zeros. Selected-gene library sizes range from 376 to 7,146, with mean 2,014.8635 and median 1,873.5. Read-only checks confirmed finite, nonnegative integer counts in both splits, identical ordered columns, unique cell IDs within splits, and no overlapping cell IDs between splits. Evaluation zero fraction was 74.0866%. The `sample_id` identifier and `cell_type` label were excluded from model inputs. No labels were used for conditioning, preprocessing, hyperparameter selection, or evaluation.

## Model and preprocessing

Selected one **negative-binomial count VAE** using the existing `analysis/train_vae.py`. Its compact latent representation and count likelihood suit sparse, overdispersed counts and the modest sample size. The encoder receives `log1p(count / cell library size * 10000)`; the negative-binomial reconstruction target remains raw counts. This per-cell transformation fits no pooled parameters. Gene ordering is preserved, and evaluation columns are explicitly aligned by training gene names. The decoder uses softmax gene proportions and learned gene-specific dispersion. Generation draws standard-normal latent vectors, resamples library sizes from training cells, and samples counts through a gamma–Poisson mixture. Realized totals fluctuate around the resampled sizes.

Settings were fixed before final evaluation: latent dimension 16, hidden width 128, batch size 128, Adam learning rate 0.001, KL weight 1, 100 epochs, seed 123, and 528 generated cells. CUDA and MPS were unavailable within the sandbox; execution completed on CPU with OMP/MKL thread limits of 2. No packages were installed. Matplotlib used a writable temporary cache after a default-cache warning.

GAN was not selected because small-sample instability and mode collapse add risk, and continuous outputs require count conversion. Flow's continuous density/count conversion is less aligned with raw counts. Diffusion adds training and sampling cost without a demonstrated advantage for this baseline. See [objective_decision.json](objective_decision.json) and [model_selection.json](model_selection.json).

## Execution and evaluation

A two-epoch smoke run completed and generated 32 cells in `vae_smoke/`; its evaluation is a pipeline check, not a competing model or checkpoint-selection criterion. The main run trained independently from the fixed seed for all 100 epochs and generated 528 cells. The supplied evaluator completed for both runs. Main training loss changed from 2137.894 to 1476.554; final evaluation loss was 1502.551. The script computes evaluation losses each epoch without gradients; these were not used for fitting, early stopping, settings selection, or choosing the checkpoint. The final epoch checkpoint was retained. These losses combine count reconstruction and KL terms and do not certify sample quality.

The following values come from [evaluation_metrics.json](vae/evaluation/evaluation_metrics.json). Real summary statistics refer to **training cells**, not the held-out split.

| Diagnostic | Real training | Synthetic |
|---|---:|---:|
| Mean selected-gene library size | 2014.8635 | 2023.5076 |
| Median selected-gene library size | 1873.5000 | 1901.0000 |
| Mean detected genes per cell | 519.9047 | 625.4205 |
| Zero fraction | 0.740048 | 0.687290 |
| Mean count per matrix entry | 1.007432 | 1.011754 |
| Mean pairwise distance | 1.102704 | 1.351348 |

Mean absolute per-gene log-normalized expression difference is **0.182950**; mean absolute per-gene detection-frequency difference is **0.065895**. These are distinct metrics, without calibrated acceptance thresholds or real-vs-real marginal-error baselines in the supplied evaluator.

Synthetic-to-training nearest-neighbor distance has mean **1.117209** and minimum **0.832332**. The held-out-to-training mean is **0.940170**. Distance calculations use log-normalized profiles, Euclidean distance divided by the square root of 2,000, at most 500 cells for within-set pairwise diversity, and at most 1,000 training references for nearest neighbors (sampling seed 123). Thus, these are sampled diagnostics rather than exhaustive privacy or duplication tests.

The library-size mean is close by construction because generation resamples training library sizes. However, synthetic cells detect about 20.3% more genes, have 5.28 percentage points fewer zeros, and have about 22.5% larger pairwise distances. The larger synthetic nearest-neighbor mean than the held-out baseline also suggests distributional mismatch. Positive sampled nearest-neighbor minima show no exact matches within the sampled reference set, but do not establish absence of memorization. Increased pairwise distance alone is not evidence of useful biological diversity.

No follow-up experiment was run: the numerical pipeline succeeded, and the fixed-budget baseline already supports a cautious recommendation. Losses were still decreasing near the end, so convergence is not established. Additional training might change results, but would not by itself validate biological structure; repeated tuning against the held-out split was avoided.

## Recommendation and limits

**IMPROVE_MODEL.** Retain the checkpoint and generated counts as an auditable baseline. Use only for software demonstrations or provisional pipeline checks with synthetic status explicit. Do not substitute these profiles for real cells in biological interpretation, differential expression, rare-cell analysis, or claims about cell-type frequencies. Do not use them as validated augmentation data or as privacy-protected replacements.

The principal observed problem is excessive gene detection and insufficient sparsity, alongside inflated profile distances. Before biological exploratory use, future work should validate gene–gene dependence, marker coexpression, population coverage, library-size distributions beyond their summaries, and task-specific utility using appropriate independent validation. These capabilities are not supplied by the existing evaluator, so they were not implemented or claimed here. No cell-type fidelity, downstream benefit, biological validity, or privacy guarantee has been demonstrated. The model lacks explicit batch/covariate effects and cell-type conditioning, and a single seed provides no uncertainty assessment. The limited selected-gene PBMC dataset also restricts generalization.

## Artifacts and reproducibility

Verified main artifacts:

- [Synthetic counts](vae/synthetic_counts.csv): 528 rows, 2,000 ordered gene columns plus a unique synthetic identifier; finite nonnegative integer counts verified.
- [VAE checkpoint](vae/vae_model.pt), [configuration](vae/training_config.json), and [100-epoch history](vae/training_history.csv).
- [Main evaluation](vae/evaluation/evaluation_metrics.json), [smoke configuration](vae_smoke/training_config.json), [smoke history](vae_smoke/training_history.csv), [smoke checkpoint](vae_smoke/vae_model.pt), [smoke counts](vae_smoke/synthetic_counts.csv), and [smoke evaluation](vae_smoke/evaluation/evaluation_metrics.json).
- Dataset profile and decision records linked above.

The main training command, from the repository root, was:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/proj2-mpl .venv/bin/python3 -u analysis/train_vae.py --data-dir data/dataset_2 --output-dir outputs/final_dataset_2/vae --epochs 100 --batch-size 128 --latent-dim 16 --hidden-dim 128 --learning-rate 0.001 --beta 1 --num-samples 528 --preprocessing log_normalize --seed 123
```

The smoke command used `vae_smoke`, 2 epochs, and 32 samples. Inspection used `analysis/inspect_data.py --data-dir data/dataset_2 --output-dir outputs/final_dataset_2`; evaluation used `analysis/evaluate_generation.py --data-dir data/dataset_2 --model-dir outputs/final_dataset_2/vae --output-dir outputs/final_dataset_2/vae/evaluation`, each with the repository Python. Choose new output directories for any rerun to preserve these results.

The complete Dataset 2 workflow was launched through run_agent.py using a single terminal command. The agent autonomously performed dataset inspection, objective interpretation, model selection, training, synthetic data generation, evaluation, and reporting. The run completed successfully with minimal manual intervention. Reproduction requires the installed and authenticated Codex CLI and the repository's Python environment. Exact model selections and training configurations may vary across independent agent runs. No source code or input datasets were changed, and report/report.pdf was not created or modified.

```bash
.venv/bin/python3 run_agent.py \
  --data-dir data/dataset_2 \
  --objective "Generate synthetic single-cell RNA-seq profiles for exploratory analysis." \
  --output-dir outputs/final_dataset_2
```