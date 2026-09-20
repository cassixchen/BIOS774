import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from analysis_utils import (
    load_train_eval_data,
    plot_embedding,
    save_embedding,
)
from preprocessing import preprocess_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--preprocessing",
        choices=["standard", "none", "log_normalize"],
        default="standard",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        train_data,
        eval_data,
        X_train,
        X_eval,
        feature_columns,
        label_columns,
    ) = load_train_eval_data(
        data_dir
    )

    X_train_processed, X_eval_processed = preprocess_data(
        X_train,
        X_eval,
        method=args.preprocessing,
    )

    pca = PCA()

    train_embedding = pca.fit_transform(
        X_train_processed
    )

    eval_embedding = pca.transform(
        X_eval_processed
    )

    explained_variance_ratio = (
        pca.explained_variance_ratio_
    )

    cumulative_variance = np.cumsum(
        explained_variance_ratio
    )

    variance_2d = float(cumulative_variance[1])

    components_90 = None

    for i, variance in enumerate(cumulative_variance):
        if variance >= 0.90:
            components_90 = i + 1
            break

    save_embedding(
        train_embedding[:, :2],
        output_dir / "pca_train.csv",
        ["PC1", "PC2"],
    )

    save_embedding(
        eval_embedding[:, :2],
        output_dir / "pca_eval.csv",
        ["PC1", "PC2"],
    )

    metrics = {
        "method": "PCA",
        "n_training_samples": len(train_data),
        "n_evaluation_samples": len(eval_data),
        "n_features": len(feature_columns),
        "n_components": 2,
        "preprocessing": args.preprocessing,
        "variance_2d": variance_2d,
        "components_for_90_percent_variance": components_90,
        "explained_variance_ratio": [
            float(value)
            for value in explained_variance_ratio
        ],
        "cumulative_explained_variance": [
            float(value)
            for value in cumulative_variance
        ],
    }

    with open(
        output_dir / "pca_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    plot_embedding(
        train_embedding[:, :2],
        train_data,
        label_columns,
        "PC1",
        "PC2",
        "PCA Projection of Training Data",
        output_dir / "pca_train_2d.png",
    )

    plot_embedding(
        eval_embedding[:, :2],
        eval_data,
        label_columns,
        "PC1",
        "PC2",
        "PCA Projection of Evaluation Data",
        output_dir / "pca_eval_2d.png",
    )

    print("PCA analysis complete.")


if __name__ == "__main__":
    main()