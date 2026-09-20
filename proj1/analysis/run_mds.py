import argparse
import json
from pathlib import Path

from sklearn.manifold import MDS

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

    X_train_processed, _ = preprocess_data(
        X_train,
        method=args.preprocessing,
    )

    mds = MDS(
        n_components=2,
        random_state=args.random_state,
    )

    train_embedding = mds.fit_transform(
        X_train_processed
    )

    save_embedding(
        train_embedding,
        output_dir / "mds_train.csv",
        ["MDS1", "MDS2"],
    )

    stress = float(
        mds.stress_
    )

    metrics = {
        "method": "mds",
        "n_training": len(train_data),
        "n_features": len(feature_columns),
        "n_components": 2,
        "random_state": args.random_state,
        "preprocessing": args.preprocessing,
        "stress": stress,
    }

    with open(
        output_dir / "mds_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    plot_embedding(
        train_embedding,
        train_data,
        label_columns,
        "MDS 1",
        "MDS 2",
        "MDS Projection of Training Data",
        output_dir / "mds_train_2d.png",
    )

    print("MDS analysis complete.")


if __name__ == "__main__":
    main()