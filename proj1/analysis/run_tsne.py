import argparse
import json
from pathlib import Path

from sklearn.manifold import TSNE

from analysis_utils import (
    load_train_data,
    plot_embedding,
    save_embedding,
)
from preprocessing import preprocess_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--learning-rate",
        default="auto",
    )

    parser.add_argument(
        "--init",
        choices=["pca", "random"],
        default="pca",
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
        X_train,
        feature_columns,
        label_columns,
    ) = load_train_data(
        data_dir
    )

    if args.learning_rate == "auto":
        learning_rate = "auto"

    else:
        learning_rate = float(args.learning_rate)

    X_train_processed, _ = preprocess_data(
        X_train,
        method=args.preprocessing,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=learning_rate,
        init=args.init,
        random_state=args.random_state,
    )

    train_embedding = tsne.fit_transform(
        X_train_processed
    )

    save_embedding(
        train_embedding,
        output_dir / "tsne_train.csv",
        ["TSNE1", "TSNE2"],
    )

    metrics = {
        "method": "t-SNE",
        "n_training_samples": len(train_data),
        "n_features": len(feature_columns),
        "n_components": 2,
        "perplexity": args.perplexity,
        "learning_rate": learning_rate,
        "init": args.init,
        "random_state": args.random_state,
        "preprocessing": args.preprocessing,
        "kl_divergence": float(tsne.kl_divergence_),
        "supports_out_of_sample_transform": False,
    }

    with open(
        output_dir / "tsne_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    plot_embedding(
        train_embedding,
        train_data,
        label_columns,
        "t-SNE 1",
        "t-SNE 2",
        "t-SNE Projection of Training Data",
        output_dir / "tsne_train_2d.png",
    )

    print("t-SNE analysis complete.")


if __name__ == "__main__":
    main()