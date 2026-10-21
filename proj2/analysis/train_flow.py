
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from analysis_utils import detect_modality
from preprocessing import preprocess_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "scrna"], default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--preprocessing", choices=["none", "standard", "log_normalize"], default="none",)
    parser.add_argument("--seed", type=int, default=3546576)
    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_image_data(path):
    df = pd.read_csv(path)
    pixel_cols = [f"pixel_{i}" for i in range(2352)]
    return df[pixel_cols].to_numpy(dtype=np.float32) / 255.0


def load_scrna_data(path, genes=None):
    df = pd.read_csv(path)

    if genes is None:
        genes = [c for c in df.columns if c not in ("sample_id", "cell_type")]

    counts = df[genes].to_numpy(dtype=np.float32)
    return counts, genes


class CouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden_dim=128):
        super().__init__()

        self.register_buffer(
            "mask", torch.tensor(mask, dtype=torch.float32)
        )
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )

    def forward(self, x, reverse=False):
        masked = x * self.mask
        scale, shift = self.network(masked).chunk(2, dim=1)

        scale = 0.8 * torch.tanh(scale) * (1 - self.mask)
        shift = shift * (1 - self.mask)

        if reverse:
            y = masked + (1 - self.mask) * (
                (x - shift) * torch.exp(-scale)
            )
            logdet = -scale.sum(dim=1)
        else:
            y = masked + (1 - self.mask) * (
                x * torch.exp(scale) + shift
            )
            logdet = scale.sum(dim=1)

        return y, logdet


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim=128, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList([
            CouplingLayer(
                dim,
                np.array([(j + i) % 2 for j in range(dim)]),
                hidden_dim,
            )
            for i in range(num_layers)
        ])
        self.dim = dim

    def forward(self, x):
        logdet = torch.zeros(len(x), device=x.device)

        for layer in self.layers:
            x, current_logdet = layer(x)
            logdet += current_logdet

        return x, logdet

    def inverse(self, z):
        for layer in reversed(self.layers):
            z, _ = layer(z, reverse=True)
        return z

    def log_prob(self, x):
        z, logdet = self.forward(x)
        log_base = -0.5 * (
            z.square() + np.log(2 * np.pi)
        ).sum(dim=1)
        return log_base + logdet

    def sample(self, n, device):
        return self.inverse(torch.randn(n, self.dim, device=device))


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total = 0.0
    n_total = 0

    with torch.set_grad_enabled(training):
        for (x,) in loader:
            x = x.to(device)
            loss = -model.log_prob(x).mean()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item() * len(x)
            n_total += len(x)

    return total / n_total


def prepare_data(args, device):
    data_dir = Path(args.data_dir)

    if args.modality == "image":
        train = load_image_data(data_dir / "train.csv")
        evaluation = load_image_data(data_dir / "eval.csv")
        genes = None
        train_libraries = None
        feature_mean = None
        feature_scale = None

    else:
        train, genes = load_scrna_data(data_dir / "train.csv")
        evaluation, _ = load_scrna_data(data_dir / "eval.csv", genes)

        train_libraries = train.sum(axis=1)

        if args.preprocessing == "standard":
            feature_mean = train.mean(axis=0, dtype=np.float64)
            feature_scale = train.std(axis=0, dtype=np.float64)
            feature_scale[feature_scale == 0] = 1.0
        else:
            feature_mean = None
            feature_scale = None

        train, evaluation = preprocess_data(
            train, evaluation, method=args.preprocessing
        )

    n_components = min(
        args.latent_dim, len(train) - 1, train.shape[1]
    )

    pca = PCA(n_components=n_components, svd_solver="full")
    train_pca = pca.fit_transform(train)
    eval_pca = pca.transform(evaluation)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_pca).astype(np.float32)
    eval_scaled = scaler.transform(eval_pca).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_scaled)),
        batch_size=args.batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(eval_scaled)),
        batch_size=args.batch_size, shuffle=False
    )

    model = RealNVP(
        n_components, args.hidden_dim, args.num_layers
    ).to(device)

    return (
        model, train_loader, eval_loader, pca, scaler,
        genes, train_libraries, feature_mean, feature_scale,
        len(train), len(evaluation)
    )


def train_model(model, train_loader, eval_loader, args, device):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    history = []

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, device, optimizer
        )
        eval_loss = run_epoch(
            model, eval_loader, device
        )

        history.append({
            "epoch": epoch + 1,
            "train_nll": train_loss,
            "eval_nll": eval_loss,
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train NLL: {train_loss:.3f} | "
            f"Eval NLL: {eval_loss:.3f}"
        )

    return history


def generate_samples(
    model, args, device, pca, scaler, genes,
    train_libraries, feature_mean, feature_scale, output_dir
):
    model.eval()

    with torch.no_grad():
        synthetic_scaled = model.sample(
            args.num_samples, device
        ).cpu().numpy()

    reconstructed = pca.inverse_transform(
        scaler.inverse_transform(synthetic_scaled)
    )

    if args.modality == "image":
        images = np.clip(
            reconstructed.reshape(-1, 28, 28, 3), 0, 1
        ).astype(np.float32)

        np.save(output_dir / "synthetic_images.npy", images)
        save_image(
            torch.from_numpy(images.transpose(0, 3, 1, 2).copy()),
            output_dir / "synthetic_grid.png",
            nrow=8,
        )

    else:
        if args.preprocessing == "log_normalize":
            expression = np.expm1(np.clip(reconstructed, 0, 15))
        elif args.preprocessing == "standard":
            expression = reconstructed * feature_scale + feature_mean
        else:
            expression = reconstructed

        expression = np.maximum(expression.astype(np.float64), 0)
        row_sums = expression.sum(axis=1, keepdims=True)
        expression[row_sums[:, 0] == 0] = 1.0
        probabilities = expression / expression.sum(axis=1, keepdims=True)

        rng = np.random.default_rng(args.seed)
        libraries = rng.choice(
            train_libraries, size=args.num_samples, replace=True
        )

        synthetic_counts = np.stack([
            rng.multinomial(
                int(round(lib)),
                np.append(
                    p[:-1] / max(p[:-1].sum(), 1.0),
                    max(0.0, 1.0 - (p[:-1] / max(p[:-1].sum(), 1.0)).sum())
                ),
            )
            for lib, p in zip(libraries, probabilities)
        ])

        synthetic_df = pd.DataFrame(synthetic_counts, columns=genes)
        synthetic_df.insert(
            0, "sample_id",
            [f"synthetic_{i + 1}" for i in range(args.num_samples)],
        )
        synthetic_df.to_csv(
            output_dir / "synthetic_counts.csv", index=False
        )


def save_results(
    model, history, args, device, pca, scaler, genes,
    feature_mean, feature_scale, n_train, n_eval, output_dir
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "modality": args.modality,
        "flow_dim": pca.n_components_,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "seed": args.seed,
    }

    if args.modality == "scrna":
        checkpoint["genes"] = genes
        checkpoint["n_genes"] = len(genes)
        checkpoint["feature_mean"] = feature_mean
        checkpoint["feature_scale"] = feature_scale

    torch.save(checkpoint, output_dir / "flow_model.pt")
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False
    )

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "pca_components": pca.n_components_,
        "pca_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "n_train": n_train,
        "n_eval": n_eval,
    })

    if args.modality == "scrna":
        config["n_genes"] = len(genes)

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()
    args.modality = args.modality or detect_modality(args.data_dir)
    set_seed(args.seed)
    device = choose_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        model, train_loader, eval_loader, pca, scaler,
        genes, train_libraries, feature_mean, feature_scale,
        n_train, n_eval
    ) = prepare_data(args, device)

    history = train_model(
        model, train_loader, eval_loader, args, device
    )

    save_results(
        model, history, args, device, pca, scaler, genes,
        feature_mean, feature_scale, n_train, n_eval, output_dir
    )

    generate_samples(
        model, args, device, pca, scaler, genes,
        train_libraries, feature_mean, feature_scale, output_dir
    )

    print("Flow analysis complete.")


if __name__ == "__main__":
    main()
