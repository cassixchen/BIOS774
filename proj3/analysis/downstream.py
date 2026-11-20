#!/usr/bin/env python3
"""Evaluate downstream paired retrieval using embeddings produced by evaluate.py."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate import cosine_similarity_stable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits


def recall_at_k(sim, k):
    n = sim.shape[0]
    k = min(k, n)
    indices = np.argpartition(sim, kth=n-k, axis=1)[:, -k:]
    return float(np.mean(np.any(indices == np.arange(n)[:, None], axis=1)))


def expanded_retrieval_metrics(sim, ks=(1, 5, 10, 20, 50, 100)):
    """Exact paired-row retrieval with conservative tie handling.

    Both modalities must be in the same paired row order. The diagonal is
    the ground-truth match. No extra model fitting is performed.
    """
    n = sim.shape[0]
    if sim.shape != (n, n) or n < 2 or not np.isfinite(sim).all():
        raise ValueError('Expected finite square similarity matrix with >=2 pairs')

    def direction(matrix):
        truth = np.diag(matrix)
        # Rank 1 means no other candidate has similarity >= the true match.
        # Counting ties conservatively avoids unstable top-k tie-breaking.
        ranks = np.count_nonzero(matrix >= truth[:, None], axis=1).astype(np.int64)
        result = {f'recall_at_{k}': float(np.mean(ranks <= min(k, n))) for k in ks}
        result.update(median_rank=float(np.median(ranks)),
                      mean_rank=float(np.mean(ranks)),
                      mean_reciprocal_rank=float(np.mean(1.0 / ranks)))
        return result, ranks

    forward, forward_ranks = direction(sim)
    backward, backward_ranks = direction(sim.T)
    mean = {f'recall_at_{k}': (forward[f'recall_at_{k}'] + backward[f'recall_at_{k}']) / 2
            for k in ks}
    mean['mean_reciprocal_rank'] = (
        forward['mean_reciprocal_rank'] + backward['mean_reciprocal_rank']) / 2
    mean['median_rank'] = (forward['median_rank'] + backward['median_rank']) / 2
    expanded = {
        'n_paired_samples': n,
        'interpretation': 'Exact paired-row matching; similar nonmatching samples count as incorrect',
        'mod1_to_mod2': forward,
        'mod2_to_mod1': backward,
        'mean_bidirectional': mean,
        'random_recall_baseline': {f'recall_at_{k}': min(k, n) / n for k in ks},
    }
    return expanded, forward_ranks, backward_ranks



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    out = Path(args.model_dir)
    cfg = json.loads((out / 'training_config.json').read_text())
    frame = pd.read_csv(out / 'eval_embeddings.csv')
    dim = int(cfg['latent_dim'])
    cols = [f'z{i}' for i in range(2 * dim)]
    if not set(cols).issubset(frame.columns):
        raise ValueError('Missing expected embedding columns; run evaluate.py first')
    if len(frame) < 2:
        raise ValueError('At least two evaluation pairs required')
    a = frame[[f'z{i}' for i in range(dim)]].to_numpy(dtype=np.float64)
    b = frame[[f'z{i}' for i in range(dim, 2 * dim)]].to_numpy(dtype=np.float64)
    sim = cosine_similarity_stable(a, b)
    metrics, forward_ranks, backward_ranks = expanded_retrieval_metrics(sim)
    metrics['source'] = str(out / 'eval_embeddings.csv')
    (out / 'downstream_quality_metrics.json').write_text(json.dumps(metrics, indent=2))
    ranks_dir = out / 'expanded_retrieval'
    ranks_dir.mkdir(exist_ok=True)
    ranks = {'mod1_to_mod2_rank': forward_ranks, 'mod2_to_mod1_rank': backward_ranks}
    if 'sample_id' in frame:
        ranks = {'sample_id': frame['sample_id'].to_numpy(), **ranks}
    pd.DataFrame(ranks).to_csv(ranks_dir / 'paired_cell_ranks.csv', index=False)
    summary_path = out / 'evaluation_metrics.json'
    summary = json.loads(summary_path.read_text())
    for k in (1, 5, 10, 20, 50, 100):
        summary[f'mod1_to_mod2_recall_at_{k}'] = metrics['mod1_to_mod2'][f'recall_at_{k}']
        summary[f'mod2_to_mod1_recall_at_{k}'] = metrics['mod2_to_mod1'][f'recall_at_{k}']
        summary[f'random_recall_at_{k}'] = metrics['random_recall_baseline'][f'recall_at_{k}']
    summary['cross_modal_recall_at_1'] = summary['mod1_to_mod2_recall_at_1']
    for k in (1, 5, 10, 20, 50, 100):
        summary[f'mean_bidirectional_recall_at_{k}'] = metrics['mean_bidirectional'][f'recall_at_{k}']
    summary['mean_bidirectional_mrr'] = metrics['mean_bidirectional']['mean_reciprocal_rank']
    summary['mean_bidirectional_median_rank'] = metrics['mean_bidirectional']['median_rank']
    summary['downstream_quality_metrics_file'] = 'downstream_quality_metrics.json'
    # Linear probing is a downstream task; fit on training embeddings only.
    train_path = out / 'train_embeddings.csv'
    if 'label' in frame.columns and train_path.is_file():
        train_frame = pd.read_csv(train_path)
        if 'label' in train_frame.columns and train_frame['label'].nunique() > 1:
            features = [f'z{i}' for i in range(2 * dim)]
            if not set(features).issubset(train_frame.columns):
                raise ValueError('Missing training embedding columns for linear probe')
            with threadpool_limits(limits=1):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(train_frame[features].to_numpy(dtype=np.float64),
                        train_frame['label'].astype(str))
                pred = clf.predict(frame[features].to_numpy(dtype=np.float64))
            y_eval = frame['label'].astype(str)
            probe = {
                'linear_probe_accuracy': float(accuracy_score(y_eval, pred)),
                'linear_probe_balanced_accuracy': float(balanced_accuracy_score(y_eval, pred)),
            }
            metrics['linear_probe'] = probe
            summary.update(probe)
            (out / 'downstream_quality_metrics.json').write_text(json.dumps(metrics, indent=2))

    if cfg['method'] == 'selfsupervised':
        summary['retrieval_interpretation'] = ('Exploratory only: within-modality autoencoder '
                                               'training does not explicitly align the two embedding spaces.')
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
