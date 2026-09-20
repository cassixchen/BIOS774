import argparse
import json
from pathlib import Path

import umap

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
        "--n-neighbors",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--metric",
        default="euclidean",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
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

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )

    train_embedding = umap_model.fit_transform(
        X_train_processed
    )

    eval_embedding = umap_model.transform(
        X_eval_processed
    )

    save_embedding(
        train_embedding,
        output_dir / "umap_train.csv",
        ["UMAP1", "UMAP2"],
    )

    save_embedding(
        eval_embedding,
        output_dir / "umap_eval.csv",
        ["UMAP1", "UMAP2"],
    )

    metrics = {
        "method": "UMAP",
        "n_training_samples": len(train_data),
        "n_evaluation_samples": len(eval_data),
        "n_features": len(feature_columns),
        "n_components": 2,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "random_state": args.random_state,
        "preprocessing": args.preprocessing,
    }

    with open(
        output_dir / "umap_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    plot_embedding(
        train_embedding,
        train_data,
        label_columns,
        "UMAP 1",
        "UMAP 2",
        "UMAP Projection of Training Data",
        output_dir / "umap_train_2d.png",
    )

    plot_embedding(
        eval_embedding,
        eval_data,
        label_columns,
        "UMAP 1",
        "UMAP 2",
        "UMAP Projection of Evaluation Data",
        output_dir / "umap_eval_2d.png",
    )

    print("UMAP analysis complete.")


if __name__ == "__main__":
    main()