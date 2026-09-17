"""
Controlled single-subject pilot for the band-attention variants.

Protocol (shared by every variant, so the comparison is paired):
  * subject 0 of the preprocessed MEEG data, Arousal 2-class label
  * trial-wise K-fold cross validation (no segment of a trial is split across
    train/test); the fold assignment only depends on the seed
  * a trial-wise split of the training trials provides the validation set, used
    only for a secondary "best-validation checkpoint" column
  * fixed epoch budget, no early stopping (a 28-42 segment validation set makes
    validation-based selection almost random)
  * TF32 disabled and deterministic cuDNN kernels, so runs are reproducible

The baseline is NOT re-evaluated here: `band_attn='none'` was verified to be
bit-identical to AT-DGNN (max abs diff 0.0), so the archived baseline runs
(experiments/results/baseline_seed*.json) pair directly with these results.

Usage
-----
    python experiments/pilot_bandattn.py --folds 3 --epochs 40 --seed 3407
"""

import argparse
import json
import os
import random
import re
import sys
import time

import h5py
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# NOTE: the repo parses sys.argv at *import* time (an unconditional set_config()
# call inside models/networks.py and train/cross_validation.py), so this script's
# argv has to be hidden while those modules are imported.
_argv = sys.argv
sys.argv = _argv[:1]
from train.cross_validation import *  # noqa: F401, E402
from models.networks import ATDGNN, ATDGNN_BandAttn  # noqa: E402
from utils.utils import get_trainable_parameter_num, LabelSmoothing  # noqa: E402
sys.argv = _argv

VARIANTS = ['AT-DGNN-BandAttn', 'BandAttn (static)', 'BandAttn (adaptive)']
SCALE_VARIANTS = ['AT-DGNN-ScaleAttn', 'ScaleAttn (static)', 'ScaleAttn (adaptive)',
                  'ScaleAttn (scalar)']


# --------------------------------------------------------------------------- #
# data (identical to the archived runs)
# --------------------------------------------------------------------------- #
def load_subject(path):
    dataset = h5py.File(path, 'r')
    return np.array(dataset['data']), np.array(dataset['label'])


def normalize(train, test):
    """Per-channel z-score using training statistics only."""
    for channel in range(train.shape[2]):
        mean = np.mean(train[:, :, channel, :])
        std = np.std(train[:, :, channel, :])
        train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
        test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
    return train, test


def prepare(idx_train, idx_test, data, label):
    data_train, label_train = data[idx_train], label[idx_train]
    data_test, label_test = data[idx_test], label[idx_test]
    data_train = np.concatenate(data_train, axis=0)
    label_train = np.concatenate(label_train, axis=0)
    if len(data_test.shape) > 4:
        data_test = np.concatenate(data_test, axis=0)
        label_test = np.concatenate(label_test, axis=0)
    data_train, data_test = normalize(data_train, data_test)
    return (torch.from_numpy(data_train).float(), torch.from_numpy(label_train).long(),
            torch.from_numpy(data_test).float(), torch.from_numpy(label_test).long())


def make_loaders(data, label, batch_size, generator, pin_memory=False):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(data, label)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator,
                      pin_memory=pin_memory)


# --------------------------------------------------------------------------- #
def build_model(name, args, idx_graph):
    common = dict(num_classes=args.num_class, input_size=tuple(args.input_shape),
                  sampling_rate=args.target_rate, num_T=args.T, out_graph=args.hidden,
                  dropout_rate=args.dropout, pool=args.pool,
                  pool_step_rate=args.pool_step_rate, idx_graph=idx_graph)
    if name == 'AT-DGNN':
        return ATDGNN(**common)
    # ---- multi-scale (frequency-scale) attention over the Tception branches ----
    if name == 'AT-DGNN-ScaleAttn':
        return ATDGNN_ScaleAttn(**common, scale_attn='both', scale_hidden=args.scale_hidden)
    if name == 'ScaleAttn (static)':
        return ATDGNN_ScaleAttn(**common, scale_attn='static', scale_hidden=args.scale_hidden)
    if name == 'ScaleAttn (adaptive)':
        return ATDGNN_ScaleAttn(**common, scale_attn='adaptive', scale_hidden=args.scale_hidden)
    if name == 'ScaleAttn (scalar)':
        return ATDGNN_ScaleAttn(**common, scale_attn='scalar', scale_hidden=args.scale_hidden)
    if name == 'AT-DGNN-BandAttn':
        return ATDGNN_BandAttn(**common, band_attn='both', band_kind=args.band_kind,
                               band_numtaps=args.band_numtaps, band_hidden=args.band_hidden,
                               band_fuse=args.band_fuse)
    if name == 'BandAttn (static)':
        return ATDGNN_BandAttn(**common, band_attn='static', band_kind=args.band_kind,
                               band_numtaps=args.band_numtaps, band_hidden=args.band_hidden,
                               band_fuse=args.band_fuse)
    if name == 'BandAttn (adaptive)':
        return ATDGNN_BandAttn(**common, band_attn='adaptive', band_kind=args.band_kind,
                               band_numtaps=args.band_numtaps, band_hidden=args.band_hidden,
                               band_fuse=args.band_fuse)
    raise ValueError(name)


def run_epoch(loader, model, loss_fn, optimizer=None, device='cpu'):
    train = optimizer is not None
    model.train(train)
    total, preds, acts = 0.0, [], []
    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        loss = loss_fn(out, yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * xb.size(0)
        preds.extend(out.argmax(1).tolist())
        acts.extend(yb.tolist())
    torch.set_grad_enabled(True)
    return total / len(loader.dataset), preds, acts


def evaluate_variant(name, data, label, args, idx_graph, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = build_model(name, args, idx_graph).to(device)
    n_params = get_trainable_parameter_num(model)
    loss_fn = LabelSmoothing(args.LS_rate) if args.LS else torch.nn.CrossEntropyLoss()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_acc, fold_f1, fold_vsel, fold_seconds = [], [], [], []

    for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
        t0 = time.time()
        rng = np.random.RandomState(args.seed + idx_fold)
        perm = rng.permutation(len(idx_train))
        n_val = max(1, int(round(len(idx_train) * args.val_rate)))
        idx_val, idx_tr = idx_train[perm[:n_val]], idx_train[perm[n_val:]]

        dtr, ltr, _, _ = prepare(idx_tr, idx_test, data, label)
        dva, lva, _, _ = prepare(idx_val, idx_test, data, label)
        dte, lte, _, _ = prepare(idx_test, idx_test, data, label)

        g = torch.Generator().manual_seed(args.seed + idx_fold)
        pin = device.type == 'cuda'
        tr_loader = make_loaders(dtr, ltr, args.batch_size, g, pin)
        va_loader = make_loaders(dva, lva, args.batch_size, g, pin)
        te_loader = make_loaders(dte, lte, args.batch_size, g, pin)

        # fresh weights for every fold
        torch.manual_seed(args.seed + idx_fold)
        model = build_model(name, args, idx_graph).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val, best_state = -1.0, None
        for _ in range(1, args.epochs + 1):
            run_epoch(tr_loader, model, loss_fn, optimizer, device=device)
            _, pv, av = run_epoch(va_loader, model, loss_fn, device=device)
            acc = accuracy_score(av, pv)
            if acc > best_val:
                best_val = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # (a) primary: the final model of the fixed budget
        _, pt, at = run_epoch(te_loader, model, loss_fn, device=device)
        final_acc, final_f1 = accuracy_score(at, pt), f1_score(at, pt, average='macro')
        if args.save_model:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'{slug}_fold{idx_fold}.pth'))
        # (b) secondary: best-validation checkpoint
        model.load_state_dict(best_state)
        _, pv2, av2 = run_epoch(te_loader, model, loss_fn, device=device)
        vsel_acc = accuracy_score(av2, pv2)

        fold_acc.append(final_acc)
        fold_f1.append(final_f1)
        fold_vsel.append(vsel_acc)
        fold_seconds.append(time.time() - t0)
        print(f'    [{name}] fold {idx_fold}: test_acc(final)={final_acc:.4f} '
              f'test_acc(best-val ckpt)={vsel_acc:.4f} test_f1(final)={final_f1:.4f} '
              f'({fold_seconds[-1]:.1f}s)', flush=True)

    return dict(variant=name, params=n_params,
                acc_mean=float(np.mean(fold_acc)), acc_std=float(np.std(fold_acc)),
                f1_mean=float(np.mean(fold_f1)), f1_std=float(np.std(fold_f1)),
                vsel_acc_mean=float(np.mean(fold_vsel)),
                vsel_acc_std=float(np.std(fold_vsel)),
                seconds=float(np.sum(fold_seconds)),
                fold_acc=[float(a) for a in fold_acc],
                fold_f1=[float(a) for a in fold_f1])


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', default='./data_eeg_MEEG_A')
    p.add_argument('--sub', type=int, default=0)
    p.add_argument('--input-shape', default='1,32,800')
    p.add_argument('--target-rate', type=int, default=200)
    p.add_argument('--num-class', type=int, default=2)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--pool', type=int, default=16)
    p.add_argument('--pool-step-rate', type=float, default=0.25)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--graph-type', default='fro')
    p.add_argument('--band-kind', default='fft', choices=['fft', 'fir'])
    p.add_argument('--band-fuse', default='residual', choices=['residual', 'replace'])
    p.add_argument('--scale-hidden', type=int, default=8)
    p.add_argument('--band-numtaps', type=int, default=129)
    p.add_argument('--band-hidden', type=int, default=8)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-rate', type=float, default=0.2)
    p.add_argument('--LS', action='store_true', default=True)
    p.add_argument('--LS-rate', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--threads', type=int, default=os.cpu_count())
    p.add_argument('--variant-set', default='band', choices=['band', 'scale'])
    p.add_argument('--variants', nargs='*', default=None)
    p.add_argument('--save-model', action='store_true', default=False)
    p.add_argument('--ckpt-dir', default='experiments/results/ckpt')
    p.add_argument('--out', default='experiments/results/bandattn_seed3407.json')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    if args.variants is None:
        args.variants = VARIANTS if args.variant_set == 'band' else SCALE_VARIANTS
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # keep the numerics of every run comparable
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.input_shape = tuple(int(v) for v in args.input_shape.split(','))
    if args.out.endswith('json'):
        m = re.search(r'seed(\d+)', args.out)
        if m and args.seed == 3407:
            args.seed = int(m.group(1))

    idx_graph = list(np.array(h5py.File(f'num_chan_local_graph_{args.graph_type}.hdf', 'r')['data']))
    data, label = load_subject(os.path.join(args.data_path, f'sub{args.sub}.hdf'))
    print(f'>>> data {data.shape} label {label.shape} '
          f'(trials={data.shape[0]}, segments/trial={data.shape[1]})')
    print(f'>>> device={device}  folds={args.folds}  epochs={args.epochs}  seed={args.seed}  '
          f'batch={args.batch_size}  band_kind={args.band_kind}')

    results = []
    for name in args.variants:
        print(f'\n=== {name} ===')
        results.append(evaluate_variant(name, data, label, args, idx_graph, device))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = dict(protocol=dict(subject=args.sub, folds=args.folds, epochs=args.epochs,
                                 batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                                 val_rate=args.val_rate, label_smoothing=args.LS_rate,
                                 band_kind=args.band_kind, band_hidden=args.band_hidden,
                                 band_fuse=args.band_fuse,
                                 early_stopping=False),
                   results=results)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print('\n=== SUMMARY ===')
    print(f'{"variant":26s} {"params":>10s} {"ACC(final)":>12s} {"F1":>11s} {"ACC(best-val)":>14s}')
    for r in results:
        print(f'{r["variant"]:26s} {r["params"]:10,d} '
              f'{r["acc_mean"] * 100:6.2f}±{r["acc_std"] * 100:4.2f} '
              f'{r["f1_mean"] * 100:5.2f}±{r["f1_std"] * 100:4.2f} '
              f'{r["vsel_acc_mean"] * 100:8.2f}±{r["vsel_acc_std"] * 100:4.2f}')
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
