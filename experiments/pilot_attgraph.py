"""
Pilot experiment: AT-DGNN + AttGraph (single subject, CPU friendly).

WHY THIS SCRIPT EXISTS
----------------------
``train/cross_validation.py`` implements the full protocol of the BIBM paper
(trial-wise 10-fold outer CV x inner 3-fold CV x a second "combine" stage).
That protocol costs 30 trainings per subject and is the right thing to run on
the full 32-subject MEEG dataset, but it is far too slow for a CPU-only pilot.

This script therefore runs a *reduced but strictly identical* protocol for every
model variant that it compares:

    outer loop : trial-wise K-fold CV (no segment of a trial is ever split
                 across train/test, exactly like the original pipeline)
    train/val  : trial-wise split of the training trials, used for early stopping
                 (no segment leakage between train and validation)
    normalisation, batch size, optimiser, label smoothing, seed, epoch budget and
    early-stopping patience are shared by all variants.

Because the protocol is shared, the *relative* comparison between variants is
meaningful. The absolute numbers are single-subject pilot numbers and must NOT
be compared with the 32-subject results reported in the papers.

Usage
-----
    python experiments/pilot_attgraph.py --sub 0 --folds 5 --epochs 60
"""

import argparse
import json
import re
import os
import random
import sys
import time

import h5py
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# NOTE: the repo parses sys.argv at *import* time (an unconditional set_config()
# call inside models/networks.py and train/cross_validation.py), so the argv of
# this script has to be hidden while those modules are imported.
_argv = sys.argv
sys.argv = _argv[:1]
from train.cross_validation import *  # noqa: F401, E402  (keeps the repo import order)
from models.networks import (ATDGNN, ATDGNN_AttGraph, ATDGNN_TemporalAttn,  # noqa: E402
                             ATDGNN_BandAttn)
from utils.utils import get_trainable_parameter_num, LabelSmoothing  # noqa: E402
sys.argv = _argv


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_subject(path):
    dataset = h5py.File(path, 'r')
    return np.array(dataset['data']), np.array(dataset['label'])


def normalize(train, test):
    """Identical to CrossValidation.normalize (per-channel z-score on train stats)."""
    for channel in range(train.shape[2]):
        mean = np.mean(train[:, :, channel, :])
        std = np.std(train[:, :, channel, :])
        train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
        test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
    return train, test


def prepare(idx_train, idx_test, data, label):
    """Identical index handling to CrossValidation.prepare_data."""
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
# model zoo used by the pilot
# --------------------------------------------------------------------------- #
def build_model(name, args, idx_graph):
    common = dict(num_classes=args.num_class, input_size=tuple(args.input_shape),
                  sampling_rate=args.target_rate, num_T=args.T, out_graph=args.hidden,
                  dropout_rate=args.dropout, pool=args.pool,
                  pool_step_rate=args.pool_step_rate, idx_graph=idx_graph)
    if name == 'AT-DGNN':
        return ATDGNN(**common)
    if name == 'AT-DGNN-AttGraph':
        return ATDGNN_AttGraph(**common, use_attn_graph=True, use_global_attn=True)
    if name == 'AttGraph (graph only)':
        return ATDGNN_AttGraph(**common, use_attn_graph=True, use_global_attn=False)
    if name == 'AttGraph (global only)':
        return ATDGNN_AttGraph(**common, use_attn_graph=False, use_global_attn=True)
    if name == 'AttGraph (param-matched)':
        return ATDGNN_AttGraph(**common, use_attn_graph=True, use_global_attn=True,
                               graph_hidden=args.graph_hidden_pm)
    if name == 'AttGraph (self-loop)':
        # diagnosis-driven fix: the measured diagonal mass of A_att is only
        # 0.07-0.14, i.e. softmax drops the self-connection that AT-DGNN adds
        # explicitly with +I.  Restore it and see whether the module recovers.
        return ATDGNN_AttGraph(**common, use_attn_graph=True, use_global_attn=True,
                               attn_self_loop=True)
    # ---- band-attention variants: sample-adaptive weighting over EEG bands ----
    if name == 'AT-DGNN-BandAttn':
        return ATDGNN_BandAttn(**common, band_attn='both')
    if name == 'BandAttn (static)':
        return ATDGNN_BandAttn(**common, band_attn='static')
    if name == 'BandAttn (adaptive)':
        return ATDGNN_BandAttn(**common, band_attn='adaptive')
    # ---- layout variants: repair the sliding-window flattening ----
    if name == 'AT-DGNN (fixed layout)':
        # dimension 1 becomes the electrode axis again, so the local filter and the
        # brain-area aggregation finally act on what they claim to act on
        return ATDGNN(**common, sliding_layout='fixed')
    if name == 'FixedLayout + TemporalAttn':
        return ATDGNN_TemporalAttn(**common, temporal_attn='factorized',
                                   temporal_placement='pre', sliding_layout='fixed')
    # ---- temporal-attention variants (AT-DGNN's own graph conv kept intact) ----
    if name == 'AT-DGNN-TemporalAttn':
        return ATDGNN_TemporalAttn(**common, temporal_attn='factorized',
                                   temporal_placement='pre')
    if name == 'TemporalAttn (post)':
        return ATDGNN_TemporalAttn(**common, temporal_attn='factorized',
                                   temporal_placement='post')
    if name == 'TemporalAttn (joint)':
        return ATDGNN_TemporalAttn(**common, temporal_attn='joint',
                                   temporal_placement='pre')
    if name == 'TemporalAttn (sharp)':
        # layer-normalised scores + learnable gain: the softmax sharpness no
        # longer depends on the magnitude of q^T W_k h
        return ATDGNN_TemporalAttn(**common, temporal_attn='sharp',
                                   temporal_placement='pre', temporal_init_gain=1.0)
    if name == 'TemporalAttn (sigmoid)':
        # unnormalised gate: the mean is free to move instead of being pinned to 1
        return ATDGNN_TemporalAttn(**common, temporal_attn='sigmoid',
                                   temporal_placement='pre')
    if name == 'TemporalAttn (residual)':
        return ATDGNN_TemporalAttn(**common, temporal_attn='factorized',
                                   temporal_placement='pre', temporal_residual=True)
    if name == 'AttGraph (self-loop, graph only)':
        return ATDGNN_AttGraph(**common, use_attn_graph=True, use_global_attn=False,
                               attn_self_loop=True)
    if name == 'AT-DGNN (width-matched)':
        # width-matched control: same graph width as the parameter-matched AttGraph,
        # needed to separate "attention mechanism" from "reduced representation width"
        return ATDGNN(**common, graph_hidden=args.graph_hidden_pm)
    raise ValueError(name)


VARIANTS = ['AT-DGNN', 'AT-DGNN-AttGraph', 'AttGraph (graph only)',
            'AttGraph (global only)', 'AttGraph (param-matched)', 'AT-DGNN (width-matched)']


# --------------------------------------------------------------------------- #
# training / evaluation
# --------------------------------------------------------------------------- #
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


def evaluate_variant(name, data, label, args, idx_graph, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = args.device_resolved
    model = build_model(name, args, idx_graph).to(device)
    n_params = get_trainable_parameter_num(model)
    loss_fn = LabelSmoothing(args.LS_rate) if args.LS else torch.nn.CrossEntropyLoss()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
    fold_final_acc, fold_final_f1 = [], []
    fold_valacc_acc, fold_valacc_f1 = [], []
    fold_best_epoch, fold_seconds = [], []

    for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
        t0 = time.time()
        # trial-wise split of the training trials into train / validation
        rng = np.random.RandomState(seed + idx_fold)
        perm = rng.permutation(len(idx_train))
        n_val = max(1, int(round(len(idx_train) * args.val_rate)))
        idx_val, idx_tr = idx_train[perm[:n_val]], idx_train[perm[n_val:]]

        dtr, ltr, _, _ = prepare(idx_tr, idx_test, data, label)
        dva, lva, _, _ = prepare(idx_val, idx_test, data, label)
        dte, lte, _, _ = prepare(idx_test, idx_test, data, label)

        g = torch.Generator().manual_seed(seed + idx_fold)
        pin = device.type == 'cuda'
        tr_loader = make_loaders(dtr, ltr, args.batch_size, g, pin)
        va_loader = make_loaders(dva, lva, args.batch_size, g, pin)
        te_loader = make_loaders(dte, lte, args.batch_size, g, pin)

        # rebuild per fold: fresh weights for every fold
        torch.manual_seed(seed + idx_fold)
        model = build_model(name, args, idx_graph).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # NOTE: with a validation set of only a handful of trials the validation
        # accuracy saturates immediately, so early stopping selects a model that
        # has seen almost no data. The pilot therefore trains every variant for
        # the *same fixed budget* and reports the final model; the best-validation
        # checkpoint is kept only as a secondary, clearly-labelled column.
        best_val, best_val_epoch, best_state = -1.0, 0, None
        for epoch in range(1, args.epochs + 1):
            run_epoch(tr_loader, model, loss_fn, optimizer, device=device)
            _, pv, av = run_epoch(va_loader, model, loss_fn, device=device)
            acc = accuracy_score(av, pv)
            if acc > best_val:
                best_val, best_val_epoch = acc, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # (a) final model after the fixed budget
        _, pt, at = run_epoch(te_loader, model, loss_fn, device=device)
        final_acc, final_f1 = accuracy_score(at, pt), f1_score(at, pt, average='macro')
        if args.save_model:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f'{slug}_fold{idx_fold}.pth'))
        # (b) secondary: best-validation checkpoint
        model.load_state_dict(best_state)
        _, pt2, at2 = run_epoch(te_loader, model, loss_fn, device=device)
        vsel_acc, vsel_f1 = accuracy_score(at2, pt2), f1_score(at2, pt2, average='macro')

        fold_final_acc.append(final_acc)
        fold_final_f1.append(final_f1)
        fold_valacc_acc.append(vsel_acc)
        fold_valacc_f1.append(vsel_f1)
        fold_best_epoch.append(best_val_epoch)
        fold_seconds.append(time.time() - t0)
        print(f'    [{name}] fold {idx_fold}: test_acc(final)={final_acc:.4f} '
              f'test_acc(best-val ckpt)={vsel_acc:.4f} '
              f'test_f1(final)={final_f1:.4f} ({fold_seconds[-1]:.1f}s)', flush=True)

    return dict(variant=name, params=n_params,
                acc_mean=float(np.mean(fold_final_acc)), acc_std=float(np.std(fold_final_acc)),
                f1_mean=float(np.mean(fold_final_f1)), f1_std=float(np.std(fold_final_f1)),
                vsel_acc_mean=float(np.mean(fold_valacc_acc)),
                vsel_acc_std=float(np.std(fold_valacc_acc)),
                vsel_f1_mean=float(np.mean(fold_valacc_f1)),
                seconds=float(np.sum(fold_seconds)),
                fold_acc=[float(a) for a in fold_final_acc],
                fold_f1=[float(f) for f in fold_final_f1])


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
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-rate', type=float, default=0.2)
    p.add_argument('--LS', action='store_true', default=True)
    p.add_argument('--LS-rate', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--graph-hidden-pm', type=int, default=600,
                   help='graph width of the parameter-matched AttGraph control')
    p.add_argument('--threads', type=int, default=os.cpu_count())
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--save-model', action='store_true', default=True,
                   help='save the final per-fold checkpoints for the attention analysis')
    p.add_argument('--ckpt-dir', default='experiments/results/ckpt')
    p.add_argument('--variants', nargs='*', default=VARIANTS)
    p.add_argument('--out', default='experiments/results/pilot_MEEG.json')
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        args.device_resolved = torch.device('cuda')
    else:
        args.device_resolved = torch.device('cpu')
    args.input_shape = tuple(int(v) for v in args.input_shape.split(','))
    # keep the numerics of every run comparable: no TF32 shortcuts, deterministic
    # cuDNN kernels (the repo sets cudnn.deterministic in train_model.set_up too)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'>>> numerics: tf32=False  cudnn.deterministic=True  '
          f'torch={torch.__version__}')
    idx_graph = list(np.array(h5py.File(f'num_chan_local_graph_{args.graph_type}.hdf', 'r')['data']))

    path = os.path.join(args.data_path, f'sub{args.sub}.hdf')
    data, label = load_subject(path)
    print(f'>>> data {data.shape} label {label.shape} '
          f'(trials={data.shape[0]}, segments/trial={data.shape[1]})')
    print(f'>>> device={args.device_resolved}'
          + (f' ({torch.cuda.get_device_name(0)})' if args.device_resolved.type == 'cuda' else '')
          + f'  threads={args.threads}  folds={args.folds}  epochs={args.epochs}  '
            f'seed={args.seed}  batch={args.batch_size}  (fixed epoch budget, no early stopping)')

    results = []
    for name in args.variants:
        print(f'\n=== {name} ===')
        results.append(evaluate_variant(name, data, label, args, idx_graph, args.seed))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = dict(protocol=dict(subject=args.sub, folds=args.folds, epochs=args.epochs,
                                 batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                                 val_rate=args.val_rate, label_smoothing=args.LS_rate,
                                 early_stopping=False),
                   results=results)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print('\n\n=== PILOT SUMMARY (single subject, reduced protocol) ===')
    print('primary = final model of the fixed-epoch budget; secondary = best-validation checkpoint')
    print(f'{"variant":26s} {"params":>9s} {"ACC(final)":>12s} {"F1(final)":>11s} '
          f'{"ACC(best-val)":>13s} {"time(s)":>8s}')
    for r in results:
        print(f'{r["variant"]:26s} {r["params"]:9d} '
              f'{r["acc_mean"]*100:6.2f}±{r["acc_std"]*100:4.2f} '
              f'{r["f1_mean"]*100:5.2f}±{r["f1_std"]*100:4.2f} '
              f'{r["vsel_acc_mean"]*100:6.2f}±{r["vsel_acc_std"]*100:4.2f} '
              f'{r["seconds"]:8.1f}')
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
