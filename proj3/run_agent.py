#!/usr/bin/env python3
"""Orchestrate Codex planning, host-side training, validation selection, and test reporting."""
import argparse
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PYTHON = str(Path(sys.executable).resolve())
METHODS = {'contrastive', 'selfsupervised', 'supervised'}
ENCODERS = {'linear', 'mlp', 'deep-mlp'}
FIELDS = {'method', 'encoder1', 'encoder2', 'epochs', 'batch_size', 'latent_dim', 'seed'}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def load(path):
    return json.loads(path.read_text())


def host(script, *args):
    cmd = [PYTHON, str(ROOT / 'analysis' / script), *map(str, args)]
    print('HOST:', shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def codex(prompt):
    subprocess.run(['codex', 'exec', '--sandbox', 'workspace-write', '-'],
                   input=prompt, text=True, cwd=ROOT, check=True)


def link_or_copy(src, dst):
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def detect(path):
    if path.is_dir() and all((path / f).is_file() for f in ('train.csv', 'eval.csv')):
        return 'prepared'
    if (path.is_dir() and (path / 'filtered_feature_bc_matrix.h5').is_file()) or (path.is_file() and path.suffix == '.h5'):
        return 'multiome'
    if path.is_dir() and all((path / f).is_file() for f in ('train_captions.csv', 'valid_captions.csv')):
        return 'rocov2'
    raise ValueError('Unrecognized dataset; use --dataset-kind to specify it')


def validate_plan(path):
    plan = load(path)
    candidates = plan.get('candidates') if set(plan) == {'candidates'} else None
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError('execution_plan.json must contain exactly three candidates')
    for c in candidates:
        if not isinstance(c, dict) or set(c) != FIELDS or c['method'] not in METHODS:
            raise ValueError(f'Invalid candidate: {c}')
        if c['encoder1'] not in ENCODERS or c['encoder2'] not in ENCODERS:
            raise ValueError(f'Invalid encoders: {c}')
        for key in ('epochs', 'batch_size', 'latent_dim', 'seed'):
            if type(c[key]) is not int or c[key] < (0 if key == 'seed' else 1):
                raise ValueError(f'Invalid {key}: {c}')
        if c['epochs'] > 1000 or c['batch_size'] > 65536 or c['latent_dim'] > 8192:
            raise ValueError('Candidate exceeds training safety limits')
    if len({json.dumps(c, sort_keys=True) for c in candidates}) != 3:
        raise ValueError('Candidates must be distinct')
    return candidates


def prepare_roco(data, prepared, out, objective):
    codex(f'''Follow AGENTS.md and .agents/skills for raw data inspection and preprocessing.
Inspect {data} and analysis/prepare_dataset_1.py --help. Compare at least two preprocessing choices.
Write {out/'raw_profile.json'} and {out/'preprocessing_decision.json'} with a 'command' string.
Objective: {objective}. Do NOT execute preprocessing or edit source files.
Use 8000 train + 1000 validation from official training, and 1000 test from official validation.
Use seed 42; host runs ResNet18 on Apple MPS outside Codex sandbox.''')
    command = load(out / 'preprocessing_decision.json').get('command')
    if not isinstance(command, str):
        raise ValueError('Preprocessing decision missing command')
    tokens = shlex.split(command)
    positions = [i for i, token in enumerate(tokens) if Path(token).name == 'prepare_dataset_1.py']
    if len(positions) != 1:
        raise ValueError('Expected prepare_dataset_1.py in preprocessing command')
    opts = tokens[positions[0] + 1:]
    allowed = {'--source', '--output', '--train-size', '--validation-size', '--eval-size',
               '--batch-size', '--seed', '--device', '--image-encoder', '--text-vectorizer',
               '--text-max-features', '--text-components', '--text-ngram-max'}
    if len(opts) % 2:
        raise ValueError('Preprocessing flags must have values')
    chosen = {}
    for flag, value in zip(opts[::2], opts[1::2]):
        if flag not in allowed or flag in chosen or value.startswith('--'):
            raise ValueError(f'Invalid preprocessing option: {flag}')
        chosen[flag] = value
    for split, minimum in (('train', 9000), ('valid', 1000)):
        frame = pd.read_csv(data / f'{split}_captions.csv', dtype={'ID': str})
        if len(frame.dropna(subset=['ID', 'Caption']).drop_duplicates('ID')) < minimum:
            raise ValueError(f'Insufficient ROCOv2 {split} examples')
    chosen.update({'--source': str(data), '--output': str(prepared), '--train-size': '8000',
                   '--validation-size': '1000', '--eval-size': '1000', '--seed': '42',
                   '--device': 'mps' if chosen.get('--image-encoder', 'resnet18') == 'resnet18' else 'cpu'})
    host('prepare_dataset_1.py', *(part for item in chosen.items() for part in item))
    meta = load(prepared / 'dataset_metadata.json')
    if tuple(meta.get(k) for k in ('train_rows', 'validation_rows', 'eval_rows')) != (8000, 1000, 1000):
        raise ValueError('ROCOv2 split sizes are incorrect')


def plan(data, prepared, out, kind, objective):
    if kind == 'rocov2':
        prepare_roco(data, prepared, out, objective)
    phase = ('ROCOv2 has already been inspected and prepared; do not redo preprocessing.'
             if kind == 'rocov2' else 'Inspect raw inputs, compare preprocessing options, and prepare numeric splits.'
             if kind == 'multiome' else 'Inputs are already prepared; skip raw inspection/preparation.')
    codex(f'''Follow AGENTS.md and relevant .agents/skills. Complete dataset inspection, objective interpretation,
preparation (if needed), and model selection. Do not edit source code or train/evaluate models.
Source: {data}; kind: {kind}; objective: {objective}; output: {out}; prepared: {prepared}.
{phase}
For Multiome use ONLY analysis/prepare_dataset_2.py, inspect --help, and use all 10970 cells if present:
8776 train, 1097 validation, 1097 held-out test; fit feature selection, IDF and SVD on train only.
For ROCOv2 use the existing 8000/1000/1000 split and train-only text transforms.
Inspect prepared features using analysis/inspect_data.py.
Write {out/'data_profile.json'}, {out/'objective_decision.json'}, {out/'model_selection.json'}.
For raw datasets also write raw_profile.json and preprocessing_decision.json (already written for ROCOv2).
Propose EXACTLY THREE distinct feasible candidates with method (contrastive/selfsupervised/supervised),
encoder1 and encoder2 (linear/mlp/deep-mlp), epochs, batch_size, latent_dim, seed.
Do not invent supervised labels. Selection is mean bidirectional validation Recall@5 only.
Write {out/'execution_plan.json'} with exactly one key 'candidates', containing exactly three objects,
each with precisely those seven candidate fields. Do not use test data for planning or selection.
HOST runs analysis/train.py, evaluate.py and downstream.py outside sandbox for MPS access.
Verify planning artifacts and prepared CSVs exist before finishing.''')
    required = ['data_profile.json', 'objective_decision.json', 'model_selection.json', 'execution_plan.json']
    if kind != 'prepared':
        required += ['raw_profile.json', 'preprocessing_decision.json']
    missing = [f for f in required if not (out / f).is_file() or not (out / f).stat().st_size]
    missing += [f for f in ('train.csv', 'eval.csv') if not (prepared / f).is_file()]
    if missing:
        raise ValueError(f'Missing planning artifacts: {missing}')
    return validate_plan(out / 'execution_plan.json')


def make_validation(prepared, out, kind, fraction, candidates):
    train = pd.read_csv(prepared / 'train.csv')
    val_file = prepared / 'validation.csv'
    selection_train = prepared / 'selection_train.csv'
    if val_file.is_file():
        if kind == 'rocov2' and selection_train.is_file():
            raise ValueError('ROCOv2 must use its dedicated 8000-row training split')
        fit = pd.read_csv(selection_train) if selection_train.is_file() else train
        val = pd.read_csv(val_file)
        isolation = 'upstream train-only fitted validation' if not selection_train.is_file() else 'upstream inner-train-only fitted validation'
    else:
        indices = list(range(len(train)))
        random.Random(42).shuffle(indices)
        n_val = max(2, round(len(train) * fraction))
        selected = set(indices[:n_val])
        fit = train.iloc[[i for i in range(len(train)) if i not in selected]]
        val = train.iloc[[i for i in range(len(train)) if i in selected]]
        isolation = 'fallback inner split: upstream preprocessing may have seen validation rows'
    if min(len(fit), len(val)) < 2 or list(fit.columns) != list(val.columns):
        raise ValueError('Invalid training/validation split')
    if kind == 'rocov2' and (len(fit), len(val)) != (8000, 1000):
        raise ValueError('ROCOv2 requires 8000/1000 selection split')
    if 'sample_id' in fit:
        ids = [set(df['sample_id']) for df in (fit, val)]
        if any(df['sample_id'].duplicated().any() for df in (fit, val)) or ids[0] & ids[1]:
            raise ValueError('Overlapping/duplicate train-validation IDs')
        test_ids = pd.read_csv(prepared / 'eval.csv', usecols=['sample_id'])['sample_id']
        if set(test_ids) & (ids[0] | ids[1]):
            raise ValueError('Test IDs overlap training/validation')
    if any(c['method'] == 'supervised' for c in candidates):
        if 'label' not in fit or fit['label'].isna().any() or fit['label'].nunique() < 2:
            raise ValueError('Supervised learning requires real, complete training labels')
    folder = out / 'model_selection_data'
    folder.mkdir()
    if val_file.is_file() and not selection_train.is_file():
        link_or_copy(prepared / 'train.csv', folder / 'train.csv')
        link_or_copy(val_file, folder / 'eval.csv')
    else:
        fit.to_csv(folder / 'train.csv', index=False)
        val.to_csv(folder / 'eval.csv', index=False)
    save(out / 'validation_split.json', {'seed': 42, 'inner_train_rows': len(fit),
         'validation_rows': len(val), 'final_training_rows': len(train),
         'selection_metric': 'mean bidirectional Recall@5', 'validation_isolation': isolation,
         'test_used_for_selection': False})
    return folder, isolation


def evaluate(data_dir, model_dir):
    host('evaluate.py', '--data-dir', data_dir, '--model-dir', model_dir)
    host('downstream.py', '--model-dir', model_dir)
    metrics = load(model_dir / 'evaluation_metrics.json')
    score = (metrics['mod1_to_mod2_recall_at_5'] + metrics['mod2_to_mod1_recall_at_5']) / 2
    if not math.isfinite(score):
        raise ValueError('Nonfinite Recall@5')
    return metrics, score


def train_candidates(candidates, selection, out, isolation):
    results = []
    for index, c in enumerate(candidates, 1):
        folder = out / 'candidate_models' / f'candidate_{index}'
        folder.mkdir(parents=True)
        opts = [part for k, v in c.items() for part in ('--' + k.replace('_', '-'), str(v))]
        host('train.py', '--data-dir', selection, '--output-dir', folder, *opts)
        metrics, score = evaluate(selection, folder)
        results.append({'candidate_index': index, 'configuration': c, 'model_directory': str(folder),
                        'validation_score': score, 'validation_metrics': metrics,
                        'device': load(folder / 'training_config.json').get('device', 'unknown')})
    winner = max(range(3), key=lambda i: results[i]['validation_score'])
    save(out / 'model_selection_results.json', {
        'selection_metric': 'mean bidirectional Recall@5', 'winner_candidate_index': winner + 1,
        'winner_configuration': candidates[winner], 'candidates': results,
        'test_used_for_selection': False, 'validation_isolation': isolation,
        'tie_break': 'earliest candidate in execution_plan.json'})
    return winner


def final_test(prepared, out, candidates, winner):
    source = out / 'candidate_models' / f'candidate_{winner + 1}'
    final = out / candidates[winner]['method']
    shutil.copytree(source, final, ignore=shutil.ignore_patterns(
        'train_embeddings.csv', 'eval_embeddings.csv', 'embedding_pca.png', 'expanded_retrieval',
        'embedding_quality_metrics.json', 'downstream_quality_metrics.json', 'evaluation_metrics.json'))
    test_data = out / 'final_test_data'
    test_data.mkdir()
    for name in ('train.csv', 'eval.csv'):
        link_or_copy(prepared / name, test_data / name)
    evaluate(test_data, final)
    for name in ('training_config.json', 'embedding_quality_metrics.json',
                 'downstream_quality_metrics.json', 'evaluation_metrics.json',
                 'expanded_retrieval/paired_cell_ranks.csv', 'eval_embeddings.csv'):
        if not (final / name).is_file():
            raise ValueError(f'Missing final result: {name}')
    # The final test embeddings are a required deliverable, not temporary output.
    embedded = pd.read_csv(final / 'eval_embeddings.csv', nrows=5)
    if 'sample_id' not in embedded.columns or not any(c.startswith('z') for c in embedded.columns):
        raise ValueError('Final embeddings must contain sample_id and learned z-columns')
    return final, test_data


def report_and_cleanup(data, prepared, out, final, objective, kind, report, temp_dirs):
    codex(f'''Follow AGENTS.md and .agents/skills to write the final markdown report at {out/report}.
Objective: {objective}; dataset: {data} ({kind}); prepared: {prepared}; output: {out}; final model: {final}.
Read all existing inspection, objective, preprocessing, model-selection, validation-split,
training and evaluation artifacts. Compare all three candidates using SAME validation split,
selection by mean bidirectional Recall@5 ONLY; distinguish untouched test evaluation.
Report representation quality (cosine gap, AUC, alignment, uniformity, effective rank) and
retrieval (Recall@1/5/10/20/50/100, MRR, median rank, random baseline).
Discuss limitations, split isolation, actual device, frozen input features vs trained encoders.
Do not run training/evaluation, edit source, invent metrics, or discuss clustering/silhouette.
Identify contrastive/eval_embeddings.csv (or the selected method's equivalent) as the saved
held-out learned representations: each row contains sample_id and concatenated modality
embeddings (first latent_dim columns = modality 1, next latent_dim = modality 2).
Include exact reproduction commands; verify report exists and is nonempty.''')
    if not (out / report).is_file() or not (out / report).stat().st_size:
        raise ValueError('Final report missing')
    for folder in temp_dirs:
        shutil.rmtree(folder)
    for candidate in (out / 'candidate_models').glob('candidate_*'):
        for name in ('train_embeddings.csv', 'eval_embeddings.csv', 'embedding_pca.png'):
            (candidate / name).unlink(missing_ok=True)
        shutil.rmtree(candidate / 'expanded_retrieval', ignore_errors=True)
    # Preserve eval_embeddings.csv (learned test representations + sample IDs).
    (final / 'train_embeddings.csv').unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--objective', required=True)
    p.add_argument('--dataset-kind', choices=['auto', 'prepared', 'rocov2', 'multiome'], default='auto')
    p.add_argument('--output-dir')
    p.add_argument('--prepared-dir')
    p.add_argument('--validation-fraction', type=float, default=0.2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    data = Path(a.data_dir).expanduser()
    data = (data if data.is_absolute() else ROOT / data).resolve()
    if not data.exists() or not a.objective.strip() or not 0.05 <= a.validation_fraction <= 0.4:
        p.error('Check dataset path, objective, and validation fraction (0.05–0.4)')
    try:
        kind = detect(data) if a.dataset_kind == 'auto' else a.dataset_kind
        suffix = {'rocov2': '1', 'multiome': '2'}.get(kind, data.name.removeprefix('dataset_'))
        out = Path(a.output_dir or f'outputs/run_dataset_{suffix}')
        out = (out if out.is_absolute() else ROOT / out).resolve()
        prepared = data if kind == 'prepared' else Path(a.prepared_dir or out / 'prepared')
        prepared = (prepared if prepared.is_absolute() else ROOT / prepared).resolve()
        if not out.is_relative_to(ROOT) or not prepared.is_relative_to(ROOT):
            raise ValueError('Output/prepared paths must be inside project')
        if kind != 'prepared' and (prepared == data or prepared == out):
            raise ValueError('Prepared directory must differ from dataset and output root')
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Output must be empty: {out}')
        if kind != 'prepared' and prepared.exists() and any(prepared.iterdir()):
            raise ValueError(f'Prepared directory must be empty: {prepared}')
        if a.dry_run:
            print(f'kind={kind}\ndata={data}\nprepared={prepared}\noutput={out}')
            return 0
        if not shutil.which('codex'):
            raise ValueError('Codex CLI not installed')
        out.mkdir(parents=True, exist_ok=True)
        candidates = plan(data, prepared, out, kind, a.objective)
        selection, isolation = make_validation(prepared, out, kind, a.validation_fraction, candidates)
        winner = train_candidates(candidates, selection, out, isolation)
        final, test_data = final_test(prepared, out, candidates, winner)
        report = f'generated_report_{suffix}.md' if suffix.isdigit() else 'generated_report.md'
        report_and_cleanup(data, prepared, out, final, a.objective, kind, report, (selection, test_data))
        print(f'Completed: {out}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f'Agent failed: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
