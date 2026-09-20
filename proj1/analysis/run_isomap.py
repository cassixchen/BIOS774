import argparse
import json
from pathlib import Path

from sklearn.manifold import Isomap

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
        "--metric",
        default="euclidean",
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

    isomap = Isomap(
        n_neighbors=args.n_neighbors,
        n_components=2,
        metric=args.metric,
    )

    train_embedding = isomap.fit_transform(
        X_train_processed
    )

    eval_embedding = isomap.transform(
        X_eval_processed
    )

    save_embedding(
        train_embedding,
        output_dir / "isomap_train.csv",
        ["Isomap1", "Isomap2"],
    )

    save_embedding(
        eval_embedding,
        output_dir / "isomap_eval.csv",
        ["Isomap1", "Isomap2"],
    )

    reconstruction_error = float(
        isomap.reconstruction_error()
    )

    metrics = {
        "method": "isomap",
        "n_training": len(train_data),
        "n_evaluation": len(eval_data),
        "n_features": len(feature_columns),
        "n_components": 2,
        "n_neighbors": args.n_neighbors,
        "metric": args.metric,
        "preprocessing": args.preprocessing,
        "reconstruction_error": reconstruction_error,
    }

    with open(
        output_dir / "isomap_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    plot_embedding(
        train_embedding,
        train_data,
        label_columns,
        "Isomap 1",
        "Isomap 2",
        "Isomap Projection of Training Data",
        output_dir / "isomap_train_2d.png",
    )

    plot_embedding(
        eval_embedding,
        eval_data,
        label_columns,
        "Isomap 1",
        "Isomap 2",
        "Isomap Projection of Evaluation Data",
        output_dir / "isomap_eval_2d.png",
    )

    print("Isomap analysis complete.")


if __name__ == "__main__":
    main()