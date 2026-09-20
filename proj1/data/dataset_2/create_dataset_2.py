from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.io import mmread


RANDOM_STATE = 123
TRAIN_FRACTION = 0.8
MIN_CELLS_PER_GENE = 3
N_GENES = 2000


def make_feature_names(selected_genes):
    feature_names = []
    seen = {}

    for gene_name in selected_genes["gene_name"]:
        name = str(gene_name)

        if name not in seen:
            seen[name] = 0
            feature_names.append(name)
        else:
            seen[name] += 1
            feature_names.append(
                f"{name}_{seen[name]}"
            )

    return feature_names


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    data_dir = Path("data/dataset_2")

    raw_dir = (
        data_dir
        / "raw"
        / "filtered_gene_bc_matrices"
        / "hg19"
    )

    matrix_path = raw_dir / "matrix.mtx"
    genes_path = raw_dir / "genes.tsv"
    barcodes_path = raw_dir / "barcodes.tsv"

    counts = mmread(matrix_path).tocsr()

    genes = pd.read_csv(
        genes_path,
        sep="\t",
        header=None,
        names=["gene_id", "gene_name"],
    )

    barcodes = pd.read_csv(
        barcodes_path,
        sep="\t",
        header=None,
        names=["barcode"],
    )

    print(
        "Original matrix shape (genes x cells):",
        counts.shape,
    )

    # Convert from genes x cells to cells x genes.
    counts = counts.T.tocsr()

    n_cells = counts.shape[0]

    indices = rng.permutation(n_cells)

    n_train = int(
        TRAIN_FRACTION * n_cells
    )

    train_indices = indices[:n_train]
    eval_indices = indices[n_train:]

    train_counts = counts[train_indices]
    eval_counts = counts[eval_indices]

    train_barcodes = barcodes.iloc[
        train_indices
    ].reset_index(drop=True)

    eval_barcodes = barcodes.iloc[
        eval_indices
    ].reset_index(drop=True)

    # Filter genes using training data only.
    cells_per_gene = np.asarray(
        (train_counts > 0).sum(axis=0)
    ).ravel()

    keep = (
        cells_per_gene >= MIN_CELLS_PER_GENE
    )

    train_filtered = train_counts[:, keep]
    eval_filtered = eval_counts[:, keep]

    filtered_genes = genes.loc[
        keep
    ].reset_index(drop=True)

    print(
        "Genes after filtering:",
        train_filtered.shape[1],
    )

    # Select highest-variance genes using training data only.
    means = np.asarray(
        train_filtered.mean(axis=0)
    ).ravel()

    squared_means = np.asarray(
        train_filtered.power(2).mean(axis=0)
    ).ravel()

    variances = squared_means - means**2

    n_select = min(
        N_GENES,
        len(variances),
    )

    selected_indices = np.argsort(
        variances
    )[-n_select:][::-1]

    train_selected = train_filtered[
        :, selected_indices
    ]

    eval_selected = eval_filtered[
        :, selected_indices
    ]

    selected_genes = filtered_genes.iloc[
        selected_indices
    ].reset_index(drop=True)

    feature_names = make_feature_names(
        selected_genes
    )

    train = pd.DataFrame(
        train_selected.toarray(),
        columns=feature_names,
    )

    eval_data = pd.DataFrame(
        eval_selected.toarray(),
        columns=feature_names,
    )

    train.insert(
        0,
        "sample_id",
        train_barcodes["barcode"].astype(str),
    )

    eval_data.insert(
        0,
        "sample_id",
        eval_barcodes["barcode"].astype(str),
    )

    train.to_csv(
        data_dir / "train.csv",
        index=False,
    )

    eval_data.to_csv(
        data_dir / "eval.csv",
        index=False,
    )

    selected_gene_table = selected_genes.copy()

    selected_gene_table["feature_name"] = (
        feature_names
    )

    selected_gene_table[
        "training_raw_variance"
    ] = variances[selected_indices]

    selected_gene_table.to_csv(
        data_dir / "selected_genes.csv",
        index=False,
    )

    schema = {
        "id_columns": ["sample_id"],
        "label_columns": [],
    }

    with open(data_dir / "schema.json", "w") as f:
        json.dump(
            schema,
            f,
            indent=4,
        )

    print("Training shape (cells x genes):", train.shape)
    print("Evaluation shape (cells x genes):", eval_data.shape)
    print("Analysis features (genes):", len(feature_names))
    print("Labels: None")
    print("Dataset 2 created.")


if __name__ == "__main__":
    main()