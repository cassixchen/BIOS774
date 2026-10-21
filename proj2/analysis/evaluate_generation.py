
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from analysis_utils import detect_modality


def save_metrics(metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "evaluation_metrics.json"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def mean_pairwise_distance(data, max_samples=500, seed=123):
    rng = np.random.default_rng(seed)

    if len(data) > max_samples:
        indices = rng.choice(
            len(data), size=max_samples, replace=False
        )
        data = data[indices]

    if len(data) < 2:
        return None

    distances = cdist(data, data, metric="euclidean")
    upper = distances[np.triu_indices(len(data), k=1)]

    return float(upper.mean() / np.sqrt(data.shape[1]))


def nearest_neighbor_distance(query, reference, max_reference=1000, seed=123):
    rng = np.random.default_rng(seed)

    if len(reference) > max_reference:
        indices = rng.choice(
            len(reference),
            size=max_reference,
            replace=False,
        )
        reference = reference[indices]

    distances = cdist(query, reference, metric="euclidean")
    nearest = distances.min(axis=1)

    return {
        "mean": float(nearest.mean() / np.sqrt(query.shape[1])),
        "minimum": float(nearest.min() / np.sqrt(query.shape[1])),
    }


def image_features(images):
    channel_means = images.mean(axis=(1, 2))
    channel_stds = images.std(axis=(1, 2))

    horizontal = np.abs(
        np.diff(images, axis=2)
    ).mean(axis=(1, 2, 3))

    vertical = np.abs(
        np.diff(images, axis=1)
    ).mean(axis=(1, 2, 3))

    return np.column_stack([
        channel_means,
        channel_stds,
        horizontal,
        vertical,
    ])


def load_real_images(path):
    df = pd.read_csv(path)
    pixel_cols = [f"pixel_{i}" for i in range(2352)]

    images = df[pixel_cols].to_numpy(dtype=np.float32)
    images = images.reshape(-1, 28, 28, 3) / 255.0

    return images


def evaluate_images(args):
    train = load_real_images(
        Path(args.data_dir) / "train.csv"
    )
    evaluation = load_real_images(
        Path(args.data_dir) / "eval.csv"
    )

    synthetic = np.load(
        Path(args.model_dir) / "synthetic_images.npy"
    )

    train_features = image_features(train)
    synthetic_features = image_features(synthetic)

    real_means = train_features.mean(axis=0)
    synthetic_means = synthetic_features.mean(axis=0)

    train_flat = train.reshape(len(train), -1)
    eval_flat = evaluation.reshape(len(evaluation), -1)
    synthetic_flat = synthetic.reshape(len(synthetic), -1)

    synthetic_nn = nearest_neighbor_distance(
        synthetic_flat, train_flat
    )
    eval_nn = nearest_neighbor_distance(
        eval_flat, train_flat
    )

    metrics = {
        "modality": "image",
        "num_train": len(train),
        "num_evaluation": len(evaluation),
        "num_synthetic": len(synthetic),
        "real_feature_means": real_means.tolist(),
        "synthetic_feature_means": synthetic_means.tolist(),
        "mean_abs_feature_difference": float(
            np.abs(real_means - synthetic_means).mean()
        ),
        "real_mean_pairwise_distance": mean_pairwise_distance(
            train_flat
        ),
        "synthetic_mean_pairwise_distance": mean_pairwise_distance(
            synthetic_flat
        ),
        "synthetic_to_train_nn_mean": synthetic_nn["mean"],
        "synthetic_to_train_nn_min": synthetic_nn["minimum"],
        "eval_to_train_nn_mean": eval_nn["mean"],
    }

    return metrics


def load_gene_counts(path, genes=None):
    df = pd.read_csv(path)

    if genes is None:
        genes = [
            col for col in df.columns
            if col not in ("sample_id", "cell_type")
        ]


    counts = df[genes].to_numpy(dtype=np.float64)


    return counts, genes


def log_normalize_counts(counts, target_sum=10000):
    libraries = counts.sum(axis=1, keepdims=True)
    safe_libraries = np.maximum(libraries, 1.0)

    return np.log1p(
        counts / safe_libraries * target_sum
    )


def gene_summary(counts):
    libraries = counts.sum(axis=1)
    detected_genes = (counts > 0).sum(axis=1)

    return {
        "mean_library_size": float(libraries.mean()),
        "median_library_size": float(np.median(libraries)),
        "mean_detected_genes_per_cell": float(
            detected_genes.mean()
        ),
        "zero_fraction": float(np.mean(counts == 0)),
        "mean_gene_count": float(counts.mean()),
    }


def evaluate_scrna(args):
    train, genes = load_gene_counts(
        Path(args.data_dir) / "train.csv"
    )
    evaluation, _ = load_gene_counts(
        Path(args.data_dir) / "eval.csv",
        genes=genes,
    )
    synthetic, _ = load_gene_counts(
        Path(args.model_dir) / "synthetic_counts.csv",
        genes=genes,
    )

    train_log = log_normalize_counts(train)
    eval_log = log_normalize_counts(evaluation)
    synthetic_log = log_normalize_counts(synthetic)

    real_gene_means = train_log.mean(axis=0)
    synthetic_gene_means = synthetic_log.mean(axis=0)

    real_detection = (train > 0).mean(axis=0)
    synthetic_detection = (synthetic > 0).mean(axis=0)

    synthetic_nn = nearest_neighbor_distance(
        synthetic_log, train_log
    )
    eval_nn = nearest_neighbor_distance(
        eval_log, train_log
    )

    metrics = {
        "modality": "scrna",
        "num_train": len(train),
        "num_evaluation": len(evaluation),
        "num_synthetic": len(synthetic),
        "num_genes": len(genes),
        "real_summary": gene_summary(train),
        "synthetic_summary": gene_summary(synthetic),
        "mean_abs_gene_expression_difference": float(
            np.abs(
                real_gene_means - synthetic_gene_means
            ).mean()
        ),
        "mean_abs_gene_detection_difference": float(
            np.abs(
                real_detection - synthetic_detection
            ).mean()
        ),
        "real_mean_pairwise_distance": mean_pairwise_distance(
            train_log
        ),
        "synthetic_mean_pairwise_distance": mean_pairwise_distance(
            synthetic_log
        ),
        "synthetic_to_train_nn_mean": synthetic_nn["mean"],
        "synthetic_to_train_nn_min": synthetic_nn["minimum"],
        "eval_to_train_nn_mean": eval_nn["mean"],
    }

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        choices=["image", "scrna"],
        default=None,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    args.modality = args.modality or detect_modality(args.data_dir)

    if args.modality == "image":
        metrics = evaluate_images(args)
    else:
        metrics = evaluate_scrna(args)

    save_metrics(metrics, args.output_dir)

    print("Generation analysis complete.")


if __name__ == "__main__":
    main()
