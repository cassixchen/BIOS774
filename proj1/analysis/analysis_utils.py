import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def get_feature_columns(train_data, schema):
    id_columns = schema.get("id_columns", [])
    label_columns = schema.get("label_columns", [])

    excluded_columns = set(id_columns + label_columns)

    feature_columns = [
        column
        for column in train_data.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(train_data[column])
    ]

    return feature_columns, label_columns


def load_train_eval_data(data_dir):
    data_dir = Path(data_dir)

    train_path = data_dir / "train.csv"
    eval_path = data_dir / "eval.csv"
    schema_path = data_dir / "schema.json"

    train_data = pd.read_csv(train_path)
    eval_data = pd.read_csv(eval_path)

    with open(schema_path) as f:
        schema = json.load(f)

    feature_columns, label_columns = get_feature_columns(
        train_data,
        schema,
    )

    X_train = train_data[feature_columns].to_numpy(dtype=float)

    X_eval = eval_data[feature_columns].to_numpy(dtype=float)

    return (
        train_data,
        eval_data,
        X_train,
        X_eval,
        feature_columns,
        label_columns,
    )


def load_train_data(data_dir):
    data_dir = Path(data_dir)

    train_path = data_dir / "train.csv"
    schema_path = data_dir / "schema.json"

    train_data = pd.read_csv(train_path)

    with open(schema_path) as f:
        schema = json.load(f)

    feature_columns, label_columns = get_feature_columns(
        train_data,
        schema,
    )

    X_train = train_data[feature_columns].to_numpy(dtype=float)

    return (
        train_data,
        X_train,
        feature_columns,
        label_columns,
    )


def save_embedding(
    embedding,
    output_path,
    column_names,
):
    embedding_df = pd.DataFrame(
        embedding,
        columns=column_names,
    )

    embedding_df.to_csv(
        output_path,
        index=False,
    )


def plot_embedding(
    embedding,
    train_data,
    label_columns,
    xlabel,
    ylabel,
    title,
    output_path,
):
    plt.figure(figsize=(8, 6))

    if label_columns:
        label_column = label_columns[0]

        for label in train_data[label_column].unique():
            mask = train_data[label_column] == label

            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=str(label),
                alpha=0.7,
            )

        plt.legend(title=label_column)

    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=0.7,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()