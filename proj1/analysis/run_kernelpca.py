import argparse
import json
from pathlib import Path

from sklearn.decomposition import KernelPCA

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
        "--kernel",
        choices=["linear", "poly", "rbf", "sigmoid", "cosine"],
        default="rbf",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
    )

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

    kernel_pca = KernelPCA(
        n_components=2,
        kernel=args.kernel,
        gamma=args.gamma,
    )

    train_embedding = kernel_pca.fit_transform(
        X_train_processed
    )

    eval_embedding = kernel_pca.transform(
        X_eval_processed
    )

    save_embedding(
        train_embedding,
        output_dir / "kernel_pca_train.csv",
        ["KPC1", "KPC2"],
    )

    save_embedding(
        eval_embedding,
        output_dir / "kernelpca_eval.csv",
        ["KPC1", "KPC2"],
    )

    metrics = {
        "method": "kernel_pca",
        "n_training": len(train_data),
        "n_evaluation": len(eval_data),
        "n_features": len(feature_columns),
        "n_components": 2,
        "kernel": args.kernel,
        "gamma": args.gamma,
        "preprocessing": args.preprocessing,
    }

    with open(
        output_dir / "kernelpca_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    plot_embedding(
        train_embedding,
        train_data,
        label_columns,
        "Kernel PC1",
        "Kernel PC2",
        "Kernel PCA Projection of Training Data",
        output_dir / "kernelpca_train_2d.png",
    )

    plot_embedding(
        eval_embedding,
        eval_data,
        label_columns,
        "Kernel PC1",
        "Kernel PC2",
        "Kernel PCA Projection of Evaluation Data",
        output_dir / "kernelpca_eval_2d.png",
    )

    print("Kernel PCA analysis complete.")


if __name__ == "__main__":
    main()