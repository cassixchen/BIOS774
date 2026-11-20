#!/usr/bin/env python3
"""Train paired-modality embeddings with selectable encoders and objectives.

'selfsupervised' means within-modality autoencoder reconstruction (MSE),
NOT masked modeling or cross-modal reconstruction.
"""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_encoder(input_dim, latent_dim, architecture):
    if architecture == 'linear':
        return nn.Linear(input_dim, latent_dim)
    if architecture == 'mlp':
        return nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
    if architecture == 'deep-mlp':
        return nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, latent_dim))
    raise ValueError(f'Unsupported encoder architecture: {architecture}')


class Net(nn.Module):
    def __init__(self, d1, d2, z, classes=0, encoder1='mlp', encoder2='mlp'):
        super().__init__()
        self.e1 = make_encoder(d1, z, encoder1)
        self.e2 = make_encoder(d2, z, encoder2)
        self.r1 = nn.Linear(z, d1)
        self.r2 = nn.Linear(z, d2)
        self.clf = nn.Linear(2*z, classes) if classes else None

    def forward(self, x, y):
        return self.e1(x), self.e2(y)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--method', choices=['contrastive', 'selfsupervised', 'supervised'], required=True,
                   help='selfsupervised = within-modality autoencoder reconstruction')
    p.add_argument('--encoder1', choices=['linear', 'mlp', 'deep-mlp'], default='mlp')
    p.add_argument('--encoder2', choices=['linear', 'mlp', 'deep-mlp'], default='mlp')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.latent_dim) < 1:
        p.error('epochs, batch-size and latent-dim must be positive')
    method = a.method
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty (do not overwrite a previous run)')
    df = pd.read_csv(Path(a.data_dir) / 'train.csv')
    cols = [[c for c in df if c.startswith(prefix + '__')] for prefix in ('mod1', 'mod2')]
    if not all(cols):
        raise ValueError('Both modalities require numeric mod1__ and mod2__ columns')
    processors, xs = [], []
    for cc in cols:
        imp = SimpleImputer(strategy='median', keep_empty_features=True)
        sc = StandardScaler()
        v = sc.fit_transform(imp.fit_transform(df[cc]))
        if not np.isfinite(v).all():
            raise ValueError('Non-finite training features after preprocessing')
        xs.append(torch.tensor(v, dtype=torch.float32))
        processors.append((imp, sc))
    labels, classes, le = None, 0, None
    if method == 'supervised':
        if 'label' not in df or df['label'].isna().any():
            raise ValueError('Supervised method requires complete genuine training labels')
        le = LabelEncoder()
        labels = torch.tensor(le.fit_transform(df['label']), dtype=torch.long)
        classes = len(le.classes_)
        if classes < 2:
            raise ValueError('Supervised learning requires at least two classes')
    device = ('cuda' if torch.cuda.is_available() else
              ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
    net = Net(len(cols[0]), len(cols[1]), a.latent_dim, classes, a.encoder1, a.encoder2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ds = TensorDataset(xs[0], xs[1], labels) if labels is not None else TensorDataset(xs[0], xs[1])
    loader = DataLoader(ds, batch_size=min(a.batch_size, len(ds)), shuffle=True, drop_last=False)
    history = []
    for epoch in range(a.epochs):
        net.train()
        losses = []
        for batch in loader:
            x, y = [t.to(device) for t in batch[:2]]
            z1, z2 = net(x, y)
            if method == 'contrastive':
                if len(x) < 2:
                    continue
                logits = (nn.functional.normalize(z1, dim=1) @
                          nn.functional.normalize(z2, dim=1).T) / 0.1
                target = torch.arange(len(x), device=device)
                loss = (nn.functional.cross_entropy(logits, target) +
                        nn.functional.cross_entropy(logits.T, target)) / 2
            elif method == 'selfsupervised':
                loss = nn.functional.mse_loss(net.r1(z1), x) + nn.functional.mse_loss(net.r2(z2), y)
            else:
                loss = nn.functional.cross_entropy(net.clf(torch.cat([z1, z2], 1)), batch[2].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if not losses:
            raise ValueError('No valid training batches')
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses))})
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), out / 'model.pt')
    joblib.dump({'processors': processors, 'columns': cols, 'label_encoder': le}, out / 'preprocessing.joblib')
    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    config = {'method': method, 'objective_description': {
        'contrastive': 'symmetric InfoNCE on paired modality embeddings',
        'selfsupervised': 'within-modality autoencoder reconstruction with mean squared error',
        'supervised': 'classification cross-entropy on concatenated embeddings'}[method],
        'encoder1': a.encoder1, 'encoder2': a.encoder2,
        'epochs': a.epochs, 'batch_size': a.batch_size, 'latent_dim': a.latent_dim,
        'seed': a.seed, 'device': device, 'dimensions': [len(c) for c in cols], 'classes': classes}
    (out / 'training_config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
