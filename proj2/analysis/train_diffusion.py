
import argparse
import json
import math
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
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
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

    return (
        pixels.reshape(-1, 28, 28, 3).transpose(0, 3, 1, 2)
        / 127.5 - 1.0
    ).copy()


def load_scrna_data(path, genes=None):
    df = pd.read_csv(path)

    if genes is None:
        genes = [c for c in df.columns if c not in ("sample_id", "cell_type")]

    counts = df[genes].to_numpy(dtype=np.float32)
    return counts, genes


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(
                half, device=t.device, dtype=torch.float32
            ) * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = t.float()[:, None] * frequencies[None, :]
        embedding = torch.cat(
            [angles.sin(), angles.cos()], dim=1
        )

        if embedding.shape[1] < self.dim:
            embedding = nn.functional.pad(
                embedding, (0, self.dim - embedding.shape[1])
            )

        return self.proj(embedding)


class ImageBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x, embedding):
        h = self.act(
            self.norm1(self.conv1(x))
            + self.time(embedding)[:, :, None, None]
        )
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class ImageDenoiser(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.time = TimeEmbedding(hidden_dim)
        self.input = nn.Conv2d(3, hidden_dim, 3, padding=1)
        self.block1 = ImageBlock(hidden_dim, hidden_dim)
        self.down = nn.Conv2d(
            hidden_dim, hidden_dim, 4, stride=2, padding=1
        )
        self.block2 = ImageBlock(hidden_dim, hidden_dim)
        self.mid = ImageBlock(hidden_dim, hidden_dim)
        self.up = nn.ConvTranspose2d(
            hidden_dim, hidden_dim, 4, stride=2, padding=1
        )
        self.block3 = ImageBlock(hidden_dim, hidden_dim)
        self.output = nn.Conv2d(hidden_dim, 3, 3, padding=1)

    def forward(self, x, t):
        embedding = self.time(t)
        skip = self.block1(self.input(x), embedding)
        h = self.block2(self.down(skip), embedding)
        h = self.mid(h, embedding)
        h = self.up(h) + skip
        return self.output(self.block3(h, embedding))


class ScrnaDenoiser(nn.Module):
    def __init__(self, n_genes, hidden_dim=512, time_dim=128):
        super().__init__()

        self.time = TimeEmbedding(time_dim)
        self.network = nn.Sequential(
            nn.Linear(n_genes + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x, t):
        return self.network(
            torch.cat([x, self.time(t)], dim=1)
        )


class DDPM:
    def __init__(self, steps, device):
        self.steps = steps
        self.betas = torch.linspace(
            1e-4, 0.02, steps, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def loss(self, model, x):
        t = torch.randint(
            0, self.steps, (len(x),), device=x.device
        )
        noise = torch.randn_like(x)
        shape = (-1,) + (1,) * (x.ndim - 1)
        alpha_bar = self.alpha_bars[t].reshape(shape)

        noisy = (
            alpha_bar.sqrt() * x
            + (1.0 - alpha_bar).sqrt() * noise
        )

        return nn.functional.mse_loss(
            model(noisy, t), noise
        )

    @torch.no_grad()
    def sample(self, model, n, sample_shape, device, batch_size):
        model.eval()
        results = []

        for start in range(0, n, batch_size):
            current = min(batch_size, n - start)
            x = torch.randn(
                (current,) + tuple(sample_shape), device=device
            )

            for step in reversed(range(self.steps)):
                t = torch.full(
                    (current,), step, device=device, dtype=torch.long
                )
                predicted_noise = model(x, t)

                alpha = self.alphas[step]
                alpha_bar = self.alpha_bars[step]

                mean = (
                    x - self.betas[step]
                    / torch.sqrt(1 - alpha_bar)
                    * predicted_noise
                ) / torch.sqrt(alpha)

                if step > 0:
                    x = mean + self.betas[step].sqrt() * torch.randn_like(x)
                else:
                    x = mean

            results.append(x.cpu().numpy())

        return np.concatenate(results, axis=0)


def run_epoch(model, diffusion, loader, device, optimizer):
    model.train()
    total = 0.0
    n_total = 0

    for (x,) in loader:
        x = x.to(device)
        loss = diffusion.loss(model, x)

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
        hidden_dim = args.hidden_dim or 64
        model = ImageDenoiser(hidden_dim).to(device)
        sample_shape = (3, 28, 28)

        genes = None
        train_libraries = None
        feature_mean = None
        feature_scale = None
        preprocessing_mean = None
        preprocessing_scale = None

    else:
        counts, genes = load_scrna_data(data_dir / "train.csv")
        train_libraries = counts.sum(axis=1)

        if args.preprocessing == "standard":
            preprocessing_mean = counts.mean(axis=0, dtype=np.float64)
            preprocessing_scale = counts.std(axis=0, dtype=np.float64)
            preprocessing_scale[preprocessing_scale == 0] = 1.0
        else:
            preprocessing_mean = None
            preprocessing_scale = None

        processed, _ = preprocess_data(
            counts, method=args.preprocessing
        )

        feature_mean = processed.mean(axis=0, dtype=np.float64)
        feature_scale = processed.std(axis=0, dtype=np.float64)
        feature_scale = np.maximum(feature_scale, 0.1)

        train = (
            (processed - feature_mean) / feature_scale
        ).astype(np.float32)

        hidden_dim = args.hidden_dim or 512
        model = ScrnaDenoiser(
            len(genes), hidden_dim
        ).to(device)
        sample_shape = (len(genes),)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    diffusion = DDPM(args.diffusion_steps, device)

    return (
        model, diffusion, train_loader, hidden_dim,
        sample_shape, genes, train_libraries,
        feature_mean, feature_scale,
        preprocessing_mean, preprocessing_scale, len(train)
    )


def train_model(model, diffusion, train_loader, args, device):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    history = []

    for epoch in range(args.epochs):
        epoch_loss = run_epoch(
            model, diffusion, train_loader, device, optimizer
        )

        history.append({
            "epoch": epoch + 1,
            "train_noise_mse": epoch_loss,
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"noise MSE: {epoch_loss:.5f}",
            flush=True,
        )

    return history


def generate_samples(
    model, diffusion, args, device, sample_shape, genes,
    train_libraries, feature_mean, feature_scale,
    preprocessing_mean, preprocessing_scale, output_dir
):
    samples = diffusion.sample(
        model, args.num_samples, sample_shape,
        device, args.batch_size
    )

    if args.modality == "image":
        images = np.clip(
            (samples + 1.0) / 2.0, 0, 1
        ).transpose(0, 2, 3, 1).astype(np.float32)

        np.save(output_dir / "synthetic_images.npy", images)
        save_image(
            torch.from_numpy(images.transpose(0, 3, 1, 2).copy()),
            output_dir / "synthetic_grid.png",
            nrow=8,
        )

    else:
        reconstructed = (
            samples.astype(np.float64) * feature_scale + feature_mean
        )

        if args.preprocessing == "log_normalize":
            expression = np.expm1(np.clip(reconstructed, 0, 15))
        elif args.preprocessing == "standard":
            expression = (
                reconstructed * preprocessing_scale + preprocessing_mean
            )
        else:
            expression = reconstructed

        expression = np.maximum(expression, 0)
        row_sums = expression.sum(axis=1)
        expression[row_sums == 0] = 1.0
        probabilities = expression / expression.sum(
            axis=1, keepdims=True
        )

        rng = np.random.default_rng(args.seed)
        libraries = rng.choice(
            train_libraries, size=args.num_samples, replace=True
        )

        synthetic_counts = []
        for lib, p in zip(libraries, probabilities):
            p = p.copy()
            subtotal = p[:-1].sum(dtype=np.float64)

            if subtotal > 1.0:
                p[:-1] /= subtotal
                subtotal = p[:-1].sum(dtype=np.float64)

            p[-1] = max(0.0, 1.0 - subtotal)
            synthetic_counts.append(
                rng.multinomial(int(round(lib)), p)
            )

        synthetic_df = pd.DataFrame(
            np.stack(synthetic_counts), columns=genes
        )
        synthetic_df.insert(
            0, "sample_id",
            [f"synthetic_{i + 1}" for i in range(args.num_samples)],
        )
        synthetic_df.to_csv(
            output_dir / "synthetic_counts.csv", index=False
        )


def save_results(
    model, history, args, device, hidden_dim, genes,
    feature_mean, feature_scale, preprocessing_mean,
    preprocessing_scale, n_train, output_dir
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "modality": args.modality,
        "hidden_dim": hidden_dim,
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
    }

    if args.modality == "scrna":
        checkpoint["genes"] = genes
        checkpoint["n_genes"] = len(genes)
        checkpoint["feature_mean"] = feature_mean
        checkpoint["feature_scale"] = feature_scale
        checkpoint["preprocessing_mean"] = preprocessing_mean
        checkpoint["preprocessing_scale"] = preprocessing_scale

    torch.save(checkpoint, output_dir / "diffusion_model.pt")
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False
    )

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "hidden_dim": hidden_dim,
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
        model, diffusion, train_loader, hidden_dim,
        sample_shape, genes, train_libraries,
        feature_mean, feature_scale,
        preprocessing_mean, preprocessing_scale, n_train
    ) = prepare_data(args, device)

    history = train_model(
        model, diffusion, train_loader, args, device
    )

    save_results(
        model, history, args, device, hidden_dim, genes,
        feature_mean, feature_scale, preprocessing_mean,
        preprocessing_scale, n_train, output_dir
    )

    generate_samples(
        model, diffusion, args, device, sample_shape, genes,
        train_libraries, feature_mean, feature_scale,
        preprocessing_mean, preprocessing_scale, output_dir
    )

    print("Diffusion analysis complete.")


if __name__ == "__main__":
    main()

