import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import get_feature_columns


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

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = data_dir / "train.csv"
    schema_path = data_dir / "schema.json"

    train_data = pd.read_csv(train_path)

    with open(schema_path) as f:
        schema = json.load(f)

    feature_columns, label_columns = get_feature_columns(
        train_data,
        schema,
    )

    id_columns = schema.get("id_columns", [])

    excluded_columns = set(
        id_columns + label_columns
    )

    all_feature_columns = [
        column
        for column in train_data.columns
        if column not in excluded_columns
    ]

    categorical_columns = [
        column
        for column in all_feature_columns
        if column not in feature_columns
    ]

    feature_data = train_data[all_feature_columns]
    numeric_data = train_data[feature_columns]

    n_samples = len(train_data)
    n_features = len(all_feature_columns)

    missing_count = int(
        feature_data.isna().sum().sum()
    )

    total_feature_values = (
        n_samples * n_features
    )

    if total_feature_values > 0:
        missing_fraction = (
            missing_count / total_feature_values
        )
    else:
        missing_fraction = 0.0

    if len(feature_columns) > 0:
        zero_count = int(
            (numeric_data == 0).sum().sum()
        )

        total_numeric_values = (
            numeric_data.shape[0]
            * numeric_data.shape[1]
        )

        zero_fraction = (
            zero_count / total_numeric_values
        )

    else:
        zero_count = 0
        zero_fraction = 0.0

    feature_summary = {}

    for column in feature_columns:
        series = numeric_data[column]

        feature_summary[column] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    row_total_summary = None

    if (
        len(feature_columns) > 0
        and not numeric_data.isna().any().any()
        and (numeric_data >= 0).all().all()
    ):
        row_totals = numeric_data.sum(axis=1)

        row_total_summary = {
            "min": float(row_totals.min()),
            "mean": float(row_totals.mean()),
            "median": float(row_totals.median()),
            "max": float(row_totals.max()),
        }

    profile = {
        "n_samples": n_samples,
        "n_features": n_features,
        "id_columns": id_columns,
        "label_columns": label_columns,
        "numeric_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "missing_count": missing_count,
        "missing_fraction": missing_fraction,
        "zero_count": zero_count,
        "zero_fraction": zero_fraction,
        "feature_summary": feature_summary,
        "row_total_summary": row_total_summary,
    }

    profile_path = (
        output_dir / "data_profile.json"
    )

    with open(profile_path, "w") as f:
        json.dump(
            profile,
            f,
            indent=4,
        )

    print("Dataset inspection complete.")

if __name__ == "__main__":
    main()