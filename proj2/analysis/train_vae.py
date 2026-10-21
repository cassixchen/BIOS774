
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from analysis_utils import detect_modality
from preprocessing import preprocess_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "scrna"], default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
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

    pixels = df[pixel_cols].to_numpy(dtype=np.float32)

    return (pixels.reshape(-1, 28, 28, 3).transpose(0, 3, 1, 2) / 255.0).copy()


def load_scrna_data(path, genes=None):
    df = pd.read_csv(path)

    if genes is None:
        genes = [c for c in df.columns if c not in ("sample_id", "cell_type")]

    counts = df[genes].to_numpy(dtype=np.float32)

    return counts, genes


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z), mu, logvar


class CountVAE(nn.Module):
    def __init__(self, n_genes, latent_dim=16, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, library_sizes):
        proportions = F.softmax(self.decoder(z), dim=1)
        mean_counts = proportions * library_sizes.unsqueeze(1)
        theta = F.softplus(self.log_theta) + 1e-4
        return mean_counts, theta

    def forward(self, x, library_sizes):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        mean_counts, theta = self.decode(z, library_sizes)
        return mean_counts, theta, mu, logvar


def image_vae_loss(reconstruction, x, mu, logvar, beta):
    reconstruction_loss = (
        (reconstruction - x).square().flatten(start_dim=1).sum(dim=1).mean()
    )
    kl_loss = -0.5 * (
        1 + logvar - mu.square() - logvar.exp()
    ).sum(dim=1).mean()

    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def negative_binomial_loss(counts, mean, theta):
    mean = mean.clamp_min(1e-8)
    theta = theta.clamp_min(1e-8)

    log_prob = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1)
        + theta * (torch.log(theta) - torch.log(theta + mean))
        + counts * (torch.log(mean) - torch.log(theta + mean))
    )

    return -log_prob.sum(dim=1).mean()


def count_vae_loss(model, encoder_x, counts, libraries, beta):
    mean, theta, mu, logvar = model(encoder_x, libraries)
    reconstruction_loss = negative_binomial_loss(counts, mean, theta)
    kl_loss = -0.5 * (
        1 + logvar - mu.square() - logvar.exp()
    ).sum(dim=1).mean()

    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def run_epoch(model, loader, device, modality, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)

    totals = np.zeros(3, dtype=float)
    n_total = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            if modality == "image":
                x = batch[0].to(device)
                reconstruction, mu, logvar = model(x)
                loss, recon, kl = image_vae_loss(
                    reconstruction, x, mu, logvar, beta
                )
                n = len(x)
            else:
                encoder_x, counts, libraries = [
                    tensor.to(device) for tensor in batch
                ]
                loss, recon, kl = count_vae_loss(
                    model, encoder_x, counts, libraries, beta
                )
                n = len(counts)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            totals += np.array([loss.item(), recon.item(), kl.item()]) * n
            n_total += n

    return totals / n_total


def prepare_data(args, device):
    data_dir = Path(args.data_dir)

    if args.modality == "image":

        latent_dim = args.latent_dim or 32
        train = load_image_data(data_dir / "train.csv")
        evaluation = load_image_data(data_dir / "eval.csv")

        train_data = TensorDataset(torch.from_numpy(train))
        eval_data = TensorDataset(torch.from_numpy(evaluation))
        model = ConvVAE(latent_dim).to(device)

        genes = None
        train_libraries = None

    else:
        latent_dim = args.latent_dim or 16
        train, genes = load_scrna_data(data_dir / "train.csv")
        evaluation, _ = load_scrna_data(data_dir / "eval.csv", genes=genes)

        train_libraries = train.sum(axis=1)
        eval_libraries = evaluation.sum(axis=1)

        train_encoder, eval_encoder = preprocess_data(
            train,
            evaluation,
            method=args.preprocessing,
        )

        train_data = TensorDataset(
            torch.from_numpy(train_encoder.astype(np.float32)),
            torch.from_numpy(train),
            torch.from_numpy(train_libraries),
        )

        eval_data = TensorDataset(
            torch.from_numpy(eval_encoder.astype(np.float32)),
            torch.from_numpy(evaluation),
            torch.from_numpy(eval_libraries),
        )

        model = CountVAE(
            n_genes=len(genes),
            latent_dim=latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False
    )

    return (
        model, train_loader, eval_loader, latent_dim,
        genes, train_libraries, len(train), len(evaluation)
    )


def train_model(model, train_loader, eval_loader, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, train_loader, device, args.modality, args.beta, optimizer
        )
        eval_metrics = run_epoch(
            model, eval_loader, device, args.modality, args.beta
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_metrics[0]),
            "train_reconstruction": float(train_metrics[1]),
            "train_kl": float(train_metrics[2]),
            "eval_loss": float(eval_metrics[0]),
            "eval_reconstruction": float(eval_metrics[1]),
            "eval_kl": float(eval_metrics[2]),
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train: {train_metrics[0]:.3f} | Eval: {eval_metrics[0]:.3f}"
        )

    return history


def generate_samples(model, args, device, latent_dim, genes, train_libraries, output_dir):
    model.eval()

    with torch.no_grad():
        z = torch.randn(args.num_samples, latent_dim, device=device)

        if args.modality == "image":
            synthetic = model.decode(z).cpu()

            save_image(synthetic, output_dir / "synthetic_grid.png", nrow=8)
            np.save(
                output_dir / "synthetic_images.npy",
                synthetic.permute(0, 2, 3, 1).numpy(),
            )
        else:
            rng = np.random.default_rng(args.seed)
            sampled_libraries = rng.choice(
                train_libraries, size=args.num_samples, replace=True
            )

            libraries = torch.tensor(
                sampled_libraries, dtype=torch.float32, device=device
            )

            means, theta = model.decode(z, libraries)
            means = means.cpu().numpy()
            theta = theta.cpu().numpy()

            gamma_rates = rng.gamma(shape=theta, scale=means / theta)
            synthetic_counts = rng.poisson(gamma_rates).astype(np.int64)

            synthetic_df = pd.DataFrame(synthetic_counts, columns=genes)
            synthetic_df.insert(
                0, "sample_id",
                [f"synthetic_{i + 1}" for i in range(args.num_samples)],
            )
            synthetic_df.to_csv(
                output_dir / "synthetic_counts.csv", index=False
            )


def save_results(
    model, history, args, device, latent_dim,
    genes, n_train, n_eval, output_dir
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "modality": args.modality,
        "latent_dim": latent_dim,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
    }

    if args.modality == "scrna":
        checkpoint["genes"] = genes
        checkpoint["n_genes"] = len(genes)

    torch.save(checkpoint, output_dir / "vae_model.pt")
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False
    )

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "latent_dim": latent_dim,
        "n_train": n_train,
        "n_eval": n_eval,
    })

    if args.modality == "scrna":
        config.update({
            "n_genes": len(genes),
            "encoder_preprocessing": args.preprocessing,
            "reconstruction_target": "raw_counts",
            "library_size_generation": "resample_training_library_sizes",
        })

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
        model, train_loader, eval_loader, latent_dim,
        genes, train_libraries, n_train, n_eval
    ) = prepare_data(args, device)

    history = train_model(
        model, train_loader, eval_loader, args, device
    )

    save_results(
        model, history, args, device, latent_dim,
        genes, n_train, n_eval, output_dir
    )

    generate_samples(
        model, args, device, latent_dim,
        genes, train_libraries, output_dir
    )

    print("VAE analysis complete.")


if __name__ == "__main__":
    main()

