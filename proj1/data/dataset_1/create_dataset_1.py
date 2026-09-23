from pathlib import Path
import json

import medmnist
import numpy as np
import pandas as pd
from medmnist import PathMNIST


RANDOM_STATE = 123
TRAIN_PER_CLASS = 500
EVAL_PER_CLASS = 100


# Use the official PathMNIST label names supplied by MedMNIST.
PATHMNIST_LABELS = {
    int(key): value
    for key, value in medmnist.INFO["pathmnist"]["label"].items()
}


def sample_by_class(images, labels, n_per_class, rng):
    labels = labels.reshape(-1)

    selected_images = []
    selected_labels = []

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]

        selected = rng.choice(
            indices,
            size=n_per_class,
            replace=False,
        )

        selected_images.append(images[selected])
        selected_labels.append(labels[selected])

    images = np.concatenate(selected_images)
    labels = np.concatenate(selected_labels)

    order = rng.permutation(len(labels))

    return images[order], labels[order]


def make_dataframe(images, labels, start_id):
    n_samples = images.shape[0]

    features = images.reshape(n_samples, -1)

    feature_columns = [
        f"pixel_{i}"
        for i in range(features.shape[1])
    ]

    data = pd.DataFrame(
        features,
        columns=feature_columns,
    )

    data.insert(
        0,
        "sample_id",
        [
            f"pathmnist_{i}"
            for i in range(
                start_id,
                start_id + n_samples,
            )
        ],
    )

    # Store the official tissue names rather than numeric class IDs.
    # This column is retained only as a label and is not an
    # analysis feature.
    data["tissue_type"] = [
        PATHMNIST_LABELS[int(label)]
        for label in labels
    ]

    return data


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    output_dir = Path("data/dataset_1")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # PathMNIST has already been downloaded manually to ~/.medmnist.
    train_dataset = PathMNIST(
        split="train"
    )

    eval_dataset = PathMNIST(
        split="test"
    )

    train_images = train_dataset.imgs
    train_labels = train_dataset.labels

    eval_images = eval_dataset.imgs
    eval_labels = eval_dataset.labels

    print("Original training images:", train_images.shape)

    print("Original evaluation images:", eval_images.shape)

    # Select the same number of observations from each class.
    train_images, train_labels = sample_by_class(
        train_images,
        train_labels,
        TRAIN_PER_CLASS,
        rng,
    )

    eval_images, eval_labels = sample_by_class(
        eval_images,
        eval_labels,
        EVAL_PER_CLASS,
        rng,
    )

    # Construct tabular datasets.
    train = make_dataframe(
        train_images,
        train_labels,
        start_id=0,
    )

    eval_data = make_dataframe(
        eval_images,
        eval_labels,
        start_id=len(train),
    )

    # Save the training and evaluation datasets.
    train.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    eval_data.to_csv(
        output_dir / "eval.csv",
        index=False,
    )

    # Tell the agent which columns are identifiers and labels.
    schema = {
        "id_columns": ["sample_id"],
        "label_columns": ["tissue_type"],
    }

    with open(
        output_dir / "schema.json",
        "w",
    ) as f:
        json.dump(
            schema,
            f,
            indent=4,
        )

    # Print a summary of the constructed dataset.
    print("Training shape:", train.shape)

    print("Evaluation shape:", eval_data.shape)

    print("Analysis features:", train.shape[1] - 2)

    print("Label: tissue_type")

    for tissue_type in sorted(
        train["tissue_type"].unique()
    ):
        print(
            f"- {tissue_type}"
        )

    print(
        "Dataset 1 created."
    )


if __name__ == "__main__":
    main()