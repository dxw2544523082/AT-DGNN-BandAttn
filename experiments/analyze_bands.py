"""
Interpretability analysis for the band-attention module.

Trains one model and reports what the module actually learned:

  * the learnable static band prior b_k and the resulting attention weights beta
    (mean over samples, spread, non-uniformity, normalised entropy)
  * the class-conditional band weights, tested at the **trial** level

Why the trial level matters
---------------------------
MEEG labels are per trial: the 14 four-second segments of one trial share the label
and are strongly correlated.  A segment-level test would treat 140 correlated
segments per class as independent samples (pseudo-replication), inflating the
effective sample size ~14x and producing spuriously tiny p-values.  The independent
unit is the trial, so the test here is n = 10 vs 10 trials, with a Bonferroni
correction over the bands.

Usage
-----
    python experiments/analyze_bands.py --train --epochs 40 --band-fuse residual
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from scipy import stats as sstats
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_argv = sys.argv
sys.argv = _argv[:1]
from train.cross_validation import *  # noqa: F401, E402
from models.networks import ATDGNN_BandAttn  # noqa: E402
from utils.utils import LabelSmoothing  # noqa: E402
sys.argv = _argv

from experiments.pilot_bandattn import (load_subject, prepare, make_loaders,  # noqa: E402
                                        run_epoch)

BAND_NAMES = ['delta (1-4)', 'theta (4-8)', 'alpha (8-13)', 'beta (13-30)', 'gamma (30-50)']


def build(args, idx_graph):
    return ATDGNN_BandAttn(
        num_classes=2, input_size=tuple(args.input_shape), sampling_rate=args.target_rate,
        num_T=args.T, out_graph=args.hidden, dropout_rate=args.dropout, pool=args.pool,
        pool_step_rate=args.pool_step_rate, idx_graph=idx_graph,
        band_attn=args.band_attn, band_kind=args.band_kind, band_fuse=args.band_fuse,
        band_numtaps=args.band_numtaps, band_hidden=args.band_hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default='./data_eeg_MEEG_A')
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--input-shape', default='1,32,800')
    ap.add_argument('--target-rate', type=int, default=200)
    ap.add_argument('--T', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--pool', type=int, default=16)
    ap.add_argument('--pool-step-rate', type=float, default=0.25)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--graph-type', default='fro')
    ap.add_argument('--band-attn', default='both', choices=['static', 'adaptive', 'both'])
    ap.add_argument('--band-kind', default='fft', choices=['fft', 'fir'])
    ap.add_argument('--band-fuse', default='residual', choices=['residual', 'replace'])
    ap.add_argument('--band-numtaps', type=int, default=129)
    ap.add_argument('--band-hidden', type=int, default=8)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--load', default='')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-rate', type=float, default=0.2)
    ap.add_argument('--LS-rate', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    ap.add_argument('--out', default='experiments/results/band_analysis.json')
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    args.input_shape = tuple(int(v) for v in args.input_shape.split(','))
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if (args.device == 'cuda' or
                                     (args.device == 'auto' and torch.cuda.is_available()))
                          else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'>>> device={device} band_attn={args.band_attn} fuse={args.band_fuse}')

    idx_graph = list(np.array(h5py.File(f'num_chan_local_graph_{args.graph_type}.hdf', 'r')['data']))
    data, label = load_subject(os.path.join(args.data_path, f'sub{args.sub}.hdf'))
    segs_per_trial = int(data.shape[1])
    model = build(args, idx_graph).to(device)

    if args.load:
        sd = torch.load(args.load, map_location=device, weights_only=False)
        sd = {k: v for k, v in sd.items() if not k.endswith('split.masks')}
        model.load_state_dict(sd, strict=False)
        print(f'    loaded {args.load}')
    elif args.train:
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(data))
        n_val = max(1, int(round(len(data) * args.val_rate)))
        idx_tr = perm[n_val:]
        dtr, ltr, _, _ = prepare(idx_tr, idx_tr, data, label)
        torch.manual_seed(args.seed)
        g = torch.Generator().manual_seed(args.seed)
        loader = make_loaders(dtr, ltr, args.batch_size, g, device.type == 'cuda')
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = LabelSmoothing(args.LS_rate)
        for ep in range(1, args.epochs + 1):
            loss, pt, at = run_epoch(loader, model, loss_fn, opt, device=device)
            if ep % 10 == 0 or ep == args.epochs:
                print(f'    epoch {ep:3d} loss={loss:.4f} train_acc={accuracy_score(at, pt):.4f}',
                      flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(os.path.dirname(args.out), 'band_model.pth'))
    else:
        raise SystemExit('pass --train or --load')

    # ---- collect beta over the whole subject, order preserved (trial ids matter) ----
    model.eval()
    d_all, l_all, _, _ = prepare(np.arange(len(data)), np.arange(len(data)), data, label)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(d_all, l_all), batch_size=args.batch_size, shuffle=False,
                        pin_memory=(device.type == 'cuda'))
    betas, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            betas.append(model.band_attention_weights(xb.to(device)).cpu().numpy())
            ys.append(yb.numpy())
    beta = np.concatenate(betas)      # (S, K)
    y = np.concatenate(ys)
    S, K = beta.shape

    # trial-level aggregation (the independent unit)
    n_trials = int(S // segs_per_trial)
    trial_id = np.arange(S) // segs_per_trial
    trial_y = np.array([int(y[t * segs_per_trial]) for t in range(n_trials)])
    beta_trial = np.stack([beta[trial_id == t].mean(0) for t in range(n_trials)])
    hi, lo = beta_trial[trial_y == 1], beta_trial[trial_y == 0]
    d_trial = hi.mean(0) - lo.mean(0)
    p_trial = np.array([sstats.ttest_ind(hi[:, k], lo[:, k], equal_var=False).pvalue
                        for k in range(K)])
    p_seg = np.array([sstats.ttest_ind(beta[y == 1, k], beta[y == 0, k],
                                       equal_var=False).pvalue for k in range(K)])
    bonf = 0.05 / K

    # beta is normalised to mean 1 across bands, so renormalise to a distribution
    # (sum = 1) before computing the entropy
    dev = np.abs(beta / beta.mean(axis=1, keepdims=True) - 1.0)
    q = beta / beta.sum(axis=1, keepdims=True)
    ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(1) / np.log(K)

    static_prior = None
    if getattr(model.band_attn, 'b', None) is not None and args.band_attn in ('static', 'both'):
        static_prior = model.band_attn.b.detach().cpu().numpy()

    print('\n=== 1) learned band weights beta (mean over samples, normalised to mean 1) ===')
    print(f'{"band":16s} {"mean":>8s} {"std":>8s} {"hi-arousal":>11s} {"lo-arousal":>11s} '
          f'{"diff":>9s} {"p(seg)":>9s} {"p(trial)":>9s}')
    rows = []
    for k in range(K):
        print(f'{BAND_NAMES[k]:16s} {beta[:, k].mean():8.4f} {beta[:, k].std():8.4f} '
              f'{hi[:, k].mean():11.4f} {lo[:, k].mean():11.4f} {d_trial[k]:+9.4f} '
              f'{p_seg[k]:9.4f} {p_trial[k]:9.4f}')
        rows.append(dict(band=BAND_NAMES[k], beta_mean=float(beta[:, k].mean()),
                         beta_std=float(beta[:, k].std()),
                         high=float(hi[:, k].mean()), low=float(lo[:, k].mean()),
                         diff_trial=float(d_trial[k]), p_segment=float(p_seg[k]),
                         p_trial=float(p_trial[k])))

    print('\n=== 2) how non-uniform are the weights? ===')
    print(f'  max |deviation from uniform| : {dev.max() * 100:.2f}%')
    print(f'  mean normalised entropy       : {ent.mean():.4f}   (1.0 = uniform, 0.0 = one-hot)')
    print(f'  most important band (mean)    : {BAND_NAMES[int(beta.mean(0).argmax())]}')
    print(f'  least important band (mean)   : {BAND_NAMES[int(beta.mean(0).argmin())]}')

    print('\n=== 3) class-conditional test, segment level vs trial level ===')
    print(f'  Bonferroni threshold for {K} bands: p < {bonf:.5f}')
    print(f'  segment level (INVALID, pseudo-replication): bands surviving = {(p_seg < bonf).sum()}/{K}')
    print(f'  trial level   (VALID, n={len(hi)} vs {len(lo)}): bands surviving = {(p_trial < bonf).sum()}/{K}')

    if static_prior is not None:
        print('\n=== 4) learnable static band prior b_k ===')
        z = static_prior - static_prior.mean()
        for k in range(K):
            print(f'  {BAND_NAMES[k]:16s} b_k = {static_prior[k]:+.4f}   (centred {z[k]:+.4f})')

    payload = dict(protocol=vars(args), n_samples=int(S), n_trials=int(n_trials),
                   bands=BAND_NAMES, beta_mean=beta.mean(0).tolist(),
                   beta_std=beta.std(0).tolist(),
                   static_prior=None if static_prior is None else static_prior.tolist(),
                   max_deviation=float(dev.max()), normalised_entropy=float(ent.mean()),
                   per_band=rows, bonferroni=float(bonf),
                   segment_level_surviving=int((p_seg < bonf).sum()),
                   trial_level_surviving=int((p_trial < bonf).sum()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
