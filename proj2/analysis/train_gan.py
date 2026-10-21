
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
from analysis_utils import detect_modality
from preprocessing import preprocess_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "scrna"], default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--preprocessing", choices=["none", "standard", "log_normalize"], default="none",)
    parser.add_argument("--seed", type=int, default=123234)
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

    return (pixels.reshape(-1, 28, 28, 3).transpose(0, 3, 1, 2) / 127.5 - 1.0).copy()


def load_scrna_data(path, genes=None):
    df = pd.read_csv(path)

    if genes is None:
        genes = [c for c in df.columns if c not in ("sample_id", "cell_type")]

    counts = df[genes].to_numpy(dtype=np.float32)

    return counts, genes


class ImageGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.network(z)


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.network(x)


class CountGenerator(nn.Module):
    def __init__(self, latent_dim, n_genes, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
            nn.Softplus(),
        )

    def forward(self, z):
        return self.network(z)


class CountDiscriminator(nn.Module):
    def __init__(self, n_genes, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_epoch(generator, discriminator, loader, optimizer_g,
                optimizer_d, latent_dim, device):
    generator.train()
    discriminator.train()

    criterion = nn.BCEWithLogitsLoss()
    g_total = 0.0
    d_total = 0.0
    n_total = 0

    for (real,) in loader:
        real = real.to(device)
        n = len(real)

        optimizer_d.zero_grad()
        z = torch.randn(n, latent_dim, device=device)
        fake = generator(z)

        real_logits = discriminator(real)
        fake_logits = discriminator(fake.detach())

        d_loss = (
            criterion(real_logits, torch.ones_like(real_logits))
            + criterion(fake_logits, torch.zeros_like(fake_logits))
        )
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        z = torch.randn(n, latent_dim, device=device)
        fake = generator(z)
        fake_logits = discriminator(fake)

        g_loss = criterion(
            fake_logits, torch.ones_like(fake_logits)
        )
        g_loss.backward()
        optimizer_g.step()

        g_total += g_loss.item() * n
        d_total += d_loss.item() * n
        n_total += n

    return g_total / n_total, d_total / n_total


def synthetic_counts_from_expression(generated, training_libraries, rng, preprocessing):
    if preprocessing == "log_normalize":
        scores = np.expm1(np.clip(generated, 0, 15))
    else:
        scores = np.maximum(generated, 0)

    proportions = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
    zero_rows = proportions.sum(axis=1) == 0
    proportions[zero_rows] = 1.0 / scores.shape[1]

    libraries = rng.choice(
        training_libraries, size=len(generated), replace=True
    )

    return np.stack([
        rng.multinomial(int(round(lib)), p)
        for lib, p in zip(libraries, proportions)
    ])


def prepare_data(args, device):
    data_dir = Path(args.data_dir)

    if args.modality == "image":
        latent_dim = args.latent_dim
        train = load_image_data(data_dir / "train.csv")

        train_data = TensorDataset(torch.from_numpy(train))
        generator = ImageGenerator(latent_dim).to(device)
        discriminator = ImageDiscriminator().to(device)

        genes = None
        train_libraries = None

    else:

        latent_dim = args.latent_dim
        train, genes = load_scrna_data(data_dir / "train.csv")

        train_libraries = train.sum(axis=1)

        train_processed, _ = preprocess_data(
            train,
            method=args.preprocessing,
        )

        train_data = TensorDataset(
            torch.from_numpy(train_processed.astype(np.float32))
        )

        generator = CountGenerator(
            latent_dim, len(genes), args.hidden_dim
        ).to(device)

        discriminator = CountDiscriminator(
            len(genes), args.hidden_dim
        ).to(device)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )

    return (
        generator, discriminator, train_loader, latent_dim,
        genes, train_libraries, len(train)
    )


def train_model(generator, discriminator, train_loader, args, device):
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    history = []

    for epoch in range(args.epochs):
        g_loss, d_loss = train_epoch(
            generator, discriminator, train_loader,
            optimizer_g, optimizer_d, args.latent_dim, device
        )

        history.append({
            "epoch": epoch + 1,
            "generator_loss": g_loss,
            "discriminator_loss": d_loss,
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"G: {g_loss:.4f} | D: {d_loss:.4f}"
        )

    return history


def generate_samples(
    generator, args, device, latent_dim, genes, train_libraries, output_dir
):
    generator.eval()

    with torch.no_grad():
        z = torch.randn(args.num_samples, latent_dim, device=device)
        generated = generator(z).cpu().numpy()

    if args.modality == "image":
        images = np.clip((generated + 1.0) / 2.0, 0.0, 1.0)

        np.save(
            output_dir / "synthetic_images.npy",
            images.transpose(0, 2, 3, 1),
        )

        save_image(
            torch.from_numpy(images),
            output_dir / "synthetic_grid.png",
            nrow=8,
        )

    else:
        rng = np.random.default_rng(args.seed)

        synthetic_counts = synthetic_counts_from_expression(
            generated, train_libraries, rng, args.preprocessing
        )

        synthetic_df = pd.DataFrame(synthetic_counts, columns=genes)
        synthetic_df.insert(
            0, "sample_id",
            [f"synthetic_{i + 1}" for i in range(args.num_samples)],
        )
        synthetic_df.to_csv(
            output_dir / "synthetic_counts.csv", index=False
        )


def save_results(
    generator, discriminator, history, args, device,
    latent_dim, genes, n_train, output_dir
):
    checkpoint = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "modality": args.modality,
        "latent_dim": latent_dim,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
    }

    if args.modality == "scrna":
        checkpoint["genes"] = genes
        checkpoint["n_genes"] = len(genes)

    torch.save(checkpoint, output_dir / "gan_model.pt")
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False
    )

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "latent_dim": latent_dim,
        "n_train": n_train,
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
        generator, discriminator, train_loader, latent_dim,
        genes, train_libraries, n_train
    ) = prepare_data(args, device)

    history = train_model(
        generator, discriminator, train_loader, args, device
    )

    save_results(
        generator, discriminator, history, args, device,
        latent_dim, genes, n_train, output_dir
    )

    generate_samples(
        generator, args, device, latent_dim,
        genes, train_libraries, output_dir
    )

    print("GAN analysis complete.")


if __name__ == "__main__":
    main()
