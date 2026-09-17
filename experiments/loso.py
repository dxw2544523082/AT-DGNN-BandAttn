"""
Leave-One-Subject-Out (LOSO) cross-subject evaluation.

Why LOSO
--------
The within-subject protocol used elsewhere in this project trains and tests on the
*same* person, and its statistical unit is a random seed that re-uses the same data -
a weak basis for inference.  LOSO instead holds out one subject at a time:

    for each subject s:  train on all other subjects, test on s

so the statistical unit is a **subject** (n = number of subjects), the folds are
genuinely independent of each other's test data, and the paired per-subject comparison
against the baseline is exactly what a reviewer expects for a cross-subject claim.

Notes on the design
-------------------
* ``--normalize per_subject`` (default) z-scores every subject with **its own**
  statistics.  This uses no labels, so it is not leakage, and it is the standard way to
  remove the huge cross-subject scale differences in EEG.  ``train_stats`` instead uses
  statistics pooled over the training subjects only.
* The epoch budget is fixed (no early stopping) for the same reason as the within-subject
  pilot: a small validation set makes validation-based selection close to random.  A
  secondary "best-validation checkpoint" number is still produced by holding out
  ``--val-subjects`` of the training subjects.
* One LOSO run already gives n = #subjects paired observations, so there is no need to
  repeat it over seeds (that would multiply the cost by the number of folds).

Usage
-----
    # quick pipeline self-test (3 subjects, 5 epochs, ~1 minute)
    python experiments/loso.py --variants "AT-DGNN" --subjects-limit 3 --epochs 5

    # full run
    python experiments/loso.py --variants "AT-DGNN" "AT-DGNN-FixedLayout" \
        --out experiments/results/loso_MEEG.json
"""

import argparse
import json
import os
import random
import sys
import time

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_argv = sys.argv
sys.argv = _argv[:1]
from train.cross_validation import *  # noqa: F401, E402
from models.networks import ATDGNN, ATDGNN_BandAttn  # noqa: E402
from utils.utils import get_trainable_parameter_num, LabelSmoothing  # noqa: E402
sys.argv = _argv


# --------------------------------------------------------------------------- #
def load_all_subjects(folder, n_subjects):
    """Load every preprocessed subject file into a list of (data, label) arrays."""
    subjects = []
    for s in range(n_subjects):
        path = os.path.join(folder, f'sub{s}.hdf')
        if not os.path.exists(path):
            break
        with h5py.File(path, 'r') as f:
            subjects.append((np.array(f['data'], dtype=np.float32),
                             np.array(f['label'])))
    return subjects


def flatten(data):
    """(trials, segs, 1, C, T) -> (trials*segs, 1, C, T)."""
    return data.reshape(-1, *data.shape[2:])


def normalize_subject(data):
    """Per-channel, per-subject z-score (no labels involved)."""
    for ch in range(data.shape[2]):
        m = data[:, :, ch, :].mean()
        s = data[:, :, ch, :].std()
        data[:, :, ch, :] = (data[:, :, ch, :] - m) / (s if s > 0 else 1.0)
    return data


def normalize_with(train_stats, data):
    for ch, (m, s) in enumerate(train_stats):
        data[:, :, ch, :] = (data[:, :, ch, :] - m) / (s if s > 0 else 1.0)
    return data


def channel_stats(data):
    return [(float(data[:, :, ch, :].mean()), float(data[:, :, ch, :].std()))
            for ch in range(data.shape[2])]


# --------------------------------------------------------------------------- #
def build_model(name, args, idx_graph):
    common = dict(num_classes=args.num_class, input_size=tuple(args.input_shape),
                  sampling_rate=args.target_rate, num_T=args.T, out_graph=args.hidden,
                  dropout_rate=args.dropout, pool=args.pool,
                  pool_step_rate=args.pool_step_rate, idx_graph=idx_graph)
    if name == 'AT-DGNN':
        return ATDGNN(**common)
    if name == 'AT-DGNN-BandAttn':
        return ATDGNN_BandAttn(**common, band_attn='both')
    if name == 'BandAttn (static)':
        return ATDGNN_BandAttn(**common, band_attn='static')
    if name == 'BandAttn (adaptive)':
        return ATDGNN_BandAttn(**common, band_attn='adaptive')
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


def make_loader(data, label, batch_size, shuffle, generator=None, pin=False):
    from torch.utils.data import DataLoader, TensorDataset
    return DataLoader(TensorDataset(data, label), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, pin_memory=pin)


# --------------------------------------------------------------------------- #
def evaluate_variant(name, subjects, args, idx_graph, device):
    """One full LOSO sweep for one model variant."""
    n_sub = len(subjects)
    model0 = build_model(name, args, idx_graph)
    n_params = get_trainable_parameter_num(model0)
    loss_fn = LabelSmoothing(args.LS_rate) if args.LS else torch.nn.CrossEntropyLoss()

    sub_acc, sub_f1, sub_vsel, sub_time = [], [], [], []
    t_start = time.time()

    for held in range(n_sub):
        t0 = time.time()
        rng = np.random.RandomState(args.seed + held)
        train_ids = [i for i in range(n_sub) if i != held]
        rng.shuffle(train_ids)
        n_val = min(args.val_subjects, max(1, len(train_ids) - 1))
        val_ids, fit_ids = train_ids[:n_val], train_ids[n_val:]

        # ---- assemble tensors (normalisation decided by --normalize) ----
        def stack(ids):
            xs = [flatten(subjects[i][0]) for i in ids]
            ls = [subjects[i][1].reshape(-1) for i in ids]
            return np.concatenate(xs, 0), np.concatenate(ls, 0)

        if args.normalize == 'train_stats':
            pooled = np.concatenate([flatten(subjects[i][0]) for i in fit_ids], 0)
            st = channel_stats(pooled)
            del pooled
            parts_x, parts_y = [], []
            for i in fit_ids + val_ids + [held]:
                x = normalize_with(st, flatten(subjects[i][0]).copy())
                parts_x.append(x)
                parts_y.append(subjects[i][1].reshape(-1))
            tr_x = np.concatenate(parts_x[:len(fit_ids)], 0)
            tr_y = np.concatenate(parts_y[:len(fit_ids)], 0)
            va_x = np.concatenate(parts_x[len(fit_ids):len(fit_ids) + len(val_ids)], 0)
            va_y = np.concatenate(parts_y[len(fit_ids):len(fit_ids) + len(val_ids)], 0)
            te_x, te_y = parts_x[-1], parts_y[-1]
        else:   # per_subject (default): each subject with its own statistics
            def one(i):
                x = normalize_subject(flatten(subjects[i][0]).copy())
                return x, subjects[i][1].reshape(-1)
            fit = [one(i) for i in fit_ids]
            val = [one(i) for i in val_ids]
            te = one(held)
            tr_x = np.concatenate([f[0] for f in fit], 0)
            tr_y = np.concatenate([f[1] for f in fit], 0)
            va_x = np.concatenate([v[0] for v in val], 0)
            va_y = np.concatenate([v[1] for v in val], 0)
            te_x, te_y = te

        to = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        tr_x, tr_y = to(tr_x), to(tr_y).long()
        va_x, va_y = to(va_x), to(va_y).long()
        te_x, te_y = to(te_x), to(te_y).long()

        g = torch.Generator().manual_seed(args.seed + held)
        tr_loader = make_loader(tr_x, tr_y, args.batch_size, True, g, device.type == 'cuda')
        va_loader = make_loader(va_x, va_y, args.batch_size, False, None, device.type == 'cuda')
        te_loader = make_loader(te_x, te_y, args.batch_size, False, None, device.type == 'cuda')

        torch.manual_seed(args.seed + held)
        model = build_model(name, args, idx_graph).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val, best_state = -1.0, None
        for ep in range(1, args.epochs + 1):
            run_epoch(tr_loader, model, loss_fn, opt, device=device)
            _, pv, av = run_epoch(va_loader, model, loss_fn, device=device)
            acc = accuracy_score(av, pv)
            if acc > best_val:
                best_val = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        _, pt, at = run_epoch(te_loader, model, loss_fn, device=device)
        acc_final, f1_final = accuracy_score(at, pt), f1_score(at, pt, average='macro')
        model.load_state_dict(best_state)
        _, pv2, av2 = run_epoch(te_loader, model, loss_fn, device=device)
        acc_vsel = accuracy_score(av2, pv2)

        sub_acc.append(acc_final)
        sub_f1.append(f1_final)
        sub_vsel.append(acc_vsel)
        sub_time.append(time.time() - t0)
        print(f'    [{name}] subject {held:2d}: test_acc(final)={acc_final:.4f} '
              f'best-val-ckpt={acc_vsel:.4f} f1={f1_final:.4f} '
              f'({sub_time[-1]:.0f}s, {len(tr_y)} train segments)', flush=True)
        del tr_x, tr_y, va_x, va_y, te_x, te_y, model, opt

    a = np.array(sub_acc)
    return dict(variant=name, params=n_params,
                acc_mean=float(a.mean()), acc_std=float(a.std()),
                f1_mean=float(np.mean(sub_f1)), f1_std=float(np.std(sub_f1)),
                vsel_acc_mean=float(np.mean(sub_vsel)),
                vsel_acc_std=float(np.std(sub_vsel)),
                seconds=float(time.time() - t_start),
                fold_acc=[float(x) for x in sub_acc],
                fold_f1=[float(x) for x in sub_f1])


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-folder', default='data_eeg_MEEG_A')
    p.add_argument('--subjects-limit', type=int, default=0,
                   help='use only the first N subjects (0 = all) for a quick self-test')
    p.add_argument('--input-shape', default='1,32,800')
    p.add_argument('--target-rate', type=int, default=200)
    p.add_argument('--num-class', type=int, default=2)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--pool', type=int, default=16)
    p.add_argument('--pool-step-rate', type=float, default=0.25)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--graph-type', default='fro')
    p.add_argument('--normalize', default='per_subject', choices=['per_subject', 'train_stats'])
    p.add_argument('--val-subjects', type=int, default=2,
                   help='how many training subjects to hold out as the validation set')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--LS', action='store_true', default=True)
    p.add_argument('--LS-rate', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--threads', type=int, default=os.cpu_count())
    p.add_argument('--variants', nargs='*',
                   default=['AT-DGNN', 'AT-DGNN-BandAttn'])
    p.add_argument('--out', default='experiments/results/loso_MEEG.json')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.input_shape = tuple(int(v) for v in args.input_shape.split(','))

    idx_graph = list(np.array(h5py.File(f'num_chan_local_graph_{args.graph_type}.hdf', 'r')['data']))
    n = args.subjects_limit if args.subjects_limit > 0 else 999
    subjects = load_all_subjects(args.data_folder, n)
    if len(subjects) < 3:
        raise SystemExit(f'need at least 3 subjects, found {len(subjects)} in {args.data_folder}')
    segs = subjects[0][0].reshape(-1, *subjects[0][0].shape[2:]).shape
    print(f'>>> LOSO: {len(subjects)} subjects, {segs} per subject, '
          f'{len(subjects[0][0])} trials x {subjects[0][0].shape[1]} segments')
    print(f'>>> device={device} epochs={args.epochs} batch={args.batch_size} '
          f'normalize={args.normalize} '
          f'val_subjects={args.val_subjects}')

    results = []
    for name in args.variants:
        print(f'\n=== LOSO: {name} ===')
        results.append(evaluate_variant(name, subjects, args, idx_graph, device))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = dict(protocol=dict(dataset=args.data_folder, n_subjects=len(subjects),
                                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                 seed=args.seed, normalize=args.normalize,
                                 val_subjects=args.val_subjects,
                                 subjects_limit=args.subjects_limit),
                   results=results)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print('\n=== LOSO SUMMARY ===')
    print(f'{"variant":28s} {"params":>10s} {"ACC over subjects":>20s} {"worst subject":>14s} {"time(s)":>9s}')
    for r in results:
        a = np.array(r['fold_acc'])
        print(f'{r["variant"]:28s} {r["params"]:10,d} '
              f'{a.mean() * 100:8.2f} ± {a.std() * 100:5.2f}    '
              f'{a.min() * 100:12.2f} {r["seconds"]:9.0f}')
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
