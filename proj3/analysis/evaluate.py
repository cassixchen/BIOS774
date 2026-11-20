#!/usr/bin/env python3
"""Evaluate learned representation quality and export embeddings for downstream tasks."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from train import Net


def cosine_similarity_stable(a, b):
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite embedding values detected')
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sim = np.einsum('ik,jk->ij', a, b, optimize=False)
    if not np.isfinite(sim).all():
        raise ValueError('Non-finite cosine similarities detected')
    return sim


def embedding_quality_metrics(a, b, sim, seed=42):
    """Held-out alignment, pair discrimination, uniformity, and effective rank.

    Cosine calculations normalize embeddings. Uniformity is estimated using
    random distinct within-modality pairs (reproducible seed).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if n < 2:
        raise ValueError('Need at least two paired samples')
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    positives = np.diag(sim)
    negatives = sim[~np.eye(n, dtype=bool)]
    rng = np.random.default_rng(seed)
    n_draw = min(100000, n * (n - 1))
    idx1 = rng.integers(0, n, size=n_draw)
    offsets = rng.integers(1, n, size=n_draw)
    idx2 = (idx1 + offsets) % n

    def uniformity(z):
        distances_squared = 2 - 2 * np.clip(np.einsum('ij,ij->i', z[idx1], z[idx2]), -1, 1)
        terms = -2 * distances_squared
        maximum = terms.max()
        return float(maximum + np.log(np.exp(terms - maximum).mean()))

    def effective_rank(z):
        centered = z - z.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = singular_values ** 2
        total = eigenvalues.sum()
        if total <= 1e-20:
            return 0.0
        probabilities = eigenvalues[eigenvalues > 0] / total
        return float(np.exp(-np.sum(probabilities * np.log(probabilities))))

    # Compare positive similarities against sampled mismatches, not against
    # the exact-pair ranking metric; this is a complementary diagnostic.
    negative_draw = negatives[rng.integers(0, len(negatives), size=n_draw)]
    positive_draw = positives[rng.integers(0, len(positives), size=n_draw)]
    pairwise_auc = float(np.mean(positive_draw > negative_draw) +
                         0.5 * np.mean(positive_draw == negative_draw))
    return {
        'n_paired_samples': int(n),
        'positive_cosine_mean': float(positives.mean()),
        'positive_cosine_std': float(positives.std()),
        'negative_cosine_mean': float(negatives.mean()),
        'negative_cosine_std': float(negatives.std()),
        'positive_negative_cosine_gap': float(positives.mean() - negatives.mean()),
        'positive_vs_negative_pairwise_auc_estimate': pairwise_auc,
        'alignment_mean_squared_distance': float(np.mean(np.sum((an - bn)**2, axis=1))),
        'uniformity_mod1': uniformity(an),
        'uniformity_mod2': uniformity(bn),
        'effective_rank_mod1': effective_rank(an),
        'effective_rank_mod2': effective_rank(bn),
        'embedding_dimension': int(a.shape[1]),
        'uniformity_and_auc_sampled_pairs': int(n_draw),
        'random_seed': seed,
        'interpretation': 'Cosine scores and alignment are cross-modal; uniformity and effective rank are within-modality. Low alignment, lower uniformity, and higher effective rank can be favorable but are not standalone proof of semantic quality.'
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir')
    p.add_argument('--model-dir')
    args = p.parse_args()
    if not args.data_dir or not args.model_dir:
        p.error('--data-dir and --model-dir are required')
    out = Path(args.model_dir)
    cfg = json.loads((out / 'training_config.json').read_text())
    prep = joblib.load(out / 'preprocessing.joblib')
    # Defaults preserve compatibility with older MLP checkpoints.
    net = Net(*cfg['dimensions'], cfg['latent_dim'], cfg['classes'],
              cfg.get('encoder1', 'mlp'), cfg.get('encoder2', 'mlp'))
    net.load_state_dict(torch.load(out / 'model.pt', map_location='cpu', weights_only=True))
    net.eval()
    embeddings = {}
    for split in ('train', 'eval'):
        df = pd.read_csv(Path(args.data_dir) / f'{split}.csv')
        tensors = []
        for columns, (imputer, scaler) in zip(prep['columns'], prep['processors']):
            transformed = scaler.transform(imputer.transform(df[columns]))
            if not np.isfinite(transformed).all():
                raise ValueError(f'Non-finite {split} input features detected')
            tensors.append(torch.tensor(transformed, dtype=torch.float32))
        with torch.no_grad():
            z1, z2 = net(*tensors)
        a1, a2 = z1.numpy(), z2.numpy()
        joint = np.concatenate((a1, a2), axis=1)
        if not np.isfinite(joint).all():
            raise ValueError(f'Non-finite {split} embeddings detected')
        embeddings[split] = (a1, a2, joint, df)
        saved = pd.DataFrame(joint, columns=[f'z{i}' for i in range(joint.shape[1])])
        if 'sample_id' in df:
            saved.insert(0, 'sample_id', df['sample_id'].to_numpy())
        if 'label' in df:
            saved['label'] = df['label'].to_numpy()
        saved.to_csv(out / f'{split}_embeddings.csv', index=False)
    a1, a2, joint, df = embeddings['eval']
    if len(df) < 2:
        raise ValueError('Evaluation split needs at least two paired samples')
    sim = cosine_similarity_stable(a1, a2)
    results = {}
    quality = embedding_quality_metrics(a1, a2, sim)
    (out / 'embedding_quality_metrics.json').write_text(json.dumps(quality, indent=2))
    results['embedding_quality_metrics_file'] = 'embedding_quality_metrics.json'
    for key in ('positive_cosine_mean', 'negative_cosine_mean',
                'positive_negative_cosine_gap', 'positive_vs_negative_pairwise_auc_estimate',
                'alignment_mean_squared_distance', 'uniformity_mod1', 'uniformity_mod2',
                'effective_rank_mod1', 'effective_rank_mod2'):
        results[key] = quality[key]
    if cfg['method'] == 'selfsupervised':
        results['retrieval_interpretation'] = 'Exploratory only: within-modality autoencoder training does not explicitly align the two embedding spaces.'
    if len(joint) >= 2:
        xy = PCA(n_components=2).fit_transform(joint.astype(np.float64))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=.7)
        ax.set(xlabel='PC1', ylabel='PC2', title='Held-out joint embeddings')
        fig.tight_layout()
        fig.savefig(out / 'embedding_pca.png', dpi=160)
        plt.close(fig)
    (out / 'evaluation_metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
