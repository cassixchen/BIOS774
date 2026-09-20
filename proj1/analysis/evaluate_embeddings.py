import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.manifold import trustworthiness

from analysis_utils import load_train_eval_data
from preprocessing import preprocess_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
    )

    return parser.parse_args()


def compute_trustworthiness(
    reference,
    embedding,
    n_neighbors,
):
    score = trustworthiness(
        reference,
        embedding,
        n_neighbors=n_neighbors,
    )

    return float(score)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    (
        train_data,
        eval_data,
        X_train,
        X_eval,
        feature_columns,
        label_columns,
    ) = load_train_eval_data(
        args.data_dir
    )

    methods = {
        "pca": {
            "train": output_dir / "pca_train.csv",
            "eval": output_dir / "pca_eval.csv",
            "metrics": output_dir / "pca_metrics.json",
        },
        "kernel_pca": {
            "train": output_dir / "kernel_pca_train.csv",
            "eval": output_dir / "kernel_pca_eval.csv",
            "metrics": output_dir / "kernel_pca_metrics.json",
        },
        "mds": {
            "train": output_dir / "mds_train.csv",
            "eval": None,
            "metrics": output_dir / "mds_metrics.json",
        },
        "isomap": {
            "train": output_dir / "isomap_train.csv",
            "eval": output_dir / "isomap_eval.csv",
            "metrics": output_dir / "isomap_metrics.json",
        },
        "lle": {
            "train": output_dir / "lle_train.csv",
            "eval": output_dir / "lle_eval.csv",
            "metrics": output_dir / "lle_metrics.json",
        },
        "laplacian_eigenmaps": {
            "train": output_dir / "laplacian_eigenmaps_train.csv",
            "eval": None,
            "metrics": output_dir / "laplacian_eigenmaps_metrics.json",
        },
        "tsne": {
            "train": output_dir / "tsne_train.csv",
            "eval": None,
            "metrics": output_dir / "tsne_metrics.json",
        },
        "umap": {
            "train": output_dir / "umap_train.csv",
            "eval": output_dir / "umap_eval.csv",
            "metrics": output_dir / "umap_metrics.json",
        },
    }

    results = {
        "metric": "trustworthiness",
        "n_neighbors": args.n_neighbors,
        "n_training_samples": len(X_train),
        "n_evaluation_samples": len(X_eval),
        "methods": {},
    }

    for method, paths in methods.items():
        if not paths["train"].exists():
            continue

        if not paths["metrics"].exists():
            continue

        with open(paths["metrics"]) as f:
            method_metrics = json.load(f)

        preprocessing = method_metrics.get(
            "preprocessing"
        )

        (
            X_train_reference,
            X_eval_reference,
        ) = preprocess_data(
            X_train,
            X_eval,
            method=preprocessing,
        )

        train_embedding = pd.read_csv(
            paths["train"]
        ).to_numpy(dtype=float)

        train_score = compute_trustworthiness(
            X_train_reference,
            train_embedding,
            args.n_neighbors,
        )

        method_result = {
            "preprocessing": preprocessing,
            "train_trustworthiness": train_score,
        }

        eval_path = paths["eval"]

        if (
            eval_path is not None
            and eval_path.exists()
        ):
            eval_embedding = pd.read_csv(
                eval_path
            ).to_numpy(dtype=float)

            eval_score = compute_trustworthiness(
                X_eval_reference,
                eval_embedding,
                args.n_neighbors,
            )

            method_result[
                "eval_trustworthiness"
            ] = eval_score

        results["methods"][method] = method_result

    evaluation_path = (
        output_dir / "embedding_evaluation.json"
    )

    with open(evaluation_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print("Embedding evaluation complete.")


if __name__ == "__main__":
    main()