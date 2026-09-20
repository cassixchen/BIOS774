from pathlib import Path
import json

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


RANDOM_STATE = 123
EVAL_SIZE = 0.2


def main():
    output_dir = Path("data/dataset_0")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    df.insert(
        0,
        "sample_id",
        [f"breast_cancer_{i}" for i in range(len(df))]
    )
    df["diagnosis"] = data.target

    train, eval_data = train_test_split(
        df,
        test_size=EVAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["diagnosis"]
    )

    train.to_csv(output_dir / "train.csv", index=False)
    eval_data.to_csv(output_dir / "eval.csv", index=False)

    schema = {
        "id_columns": ["sample_id"],
        "label_columns": ["diagnosis"]
    }

    with open(output_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=4)

    print("Training shape:", train.shape)
    print("Evaluation shape:", eval_data.shape)
    print("Analysis features:", len(data.feature_names))
    print("Labels:")
    for i, label in enumerate(data.target_names):
        print(f"  {i}: {label}")

    print("Dataset 0 created.")


if __name__ == "__main__":
    main()