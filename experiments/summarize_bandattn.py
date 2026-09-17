"""
Paired summary of the band-attention pilot.

Merges the archived baseline runs and the band-attention runs by seed (both use
identical fold splits for a given seed, so the comparison is paired) and prints
the tables used in the paper.

The baseline resweep is unnecessary: `band_attn='none'` is bit-identical to
AT-DGNN (max abs diff 0.0), so experiments/results/baseline_seed*.json pair
directly with experiments/results/bandattn_seed*.json.

Usage
-----
    python experiments/summarize_bandattn.py
"""

import argparse
import glob
import json
import sys

import numpy as np

BASELINE = 'AT-DGNN'
ORDER = [BASELINE, 'AT-DGNN-BandAttn', 'BandAttn (static)', 'BandAttn (adaptive)']


def load(patterns):
    per_seed, protocol = {}, None
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding='utf-8') as f:
                payload = json.load(f)
            protocol = protocol or payload['protocol']
            seed = payload['protocol']['seed']
            per_seed.setdefault(seed, {}).update({r['variant']: r for r in payload['results']})
    return per_seed, protocol


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='experiments/results/baseline_seed*.json')
    ap.add_argument('--runs', default='experiments/results/bandattn_seed*.json')
    ap.add_argument('--collapse-threshold', type=float, default=0.70)
    ap.add_argument('--show-per-seed', action='store_true', default=True)
    args = ap.parse_args()

    per_seed, protocol = load([args.baseline, args.runs])
    present = {v for s in per_seed for v in per_seed[s]}
    wanted = [v for v in ORDER if v in present]
    # keep the seeds on which *every* wanted variant was evaluated (the intersection)
    seeds = [s for s in sorted(per_seed) if all(v in per_seed[s] for v in wanted)]
    variants = wanted
    if BASELINE not in variants or len(variants) < 2 or len(seeds) < 2:
        print('need the baseline plus at least one band-attention variant on >=2 shared seeds; '
              f'found seeds={len(seeds)} variants={sorted(present)}')
        return
    n = len(seeds)
    base = np.array([np.mean(per_seed[s][BASELINE]['fold_acc']) for s in seeds])

    print('# Band-attention pilot — paired summary\n')
    print(f'seeds: {seeds}  (n = {n}; every variant sees the same folds per seed)\n')
    print('| protocol | value |')
    print('|---|---|')
    for k, v in protocol.items():
        print(f'| {k} | {v} |')

    print(f'\n## Table 1 — accuracy across seeds (n = {n})\n')
    print('| Model | Params | ACC over seeds (%) | ACC over folds (%) | worst fold (%) | '
          'collapse rate | ACC (best-val ckpt, %) |')
    print('|---|---|---|---|---|---|---|')
    stats = {}
    for v in variants:
        sm = np.array([np.mean(per_seed[s][v]['fold_acc']) for s in seeds])
        fl = np.array([f for s in seeds for f in per_seed[s][v]['fold_acc']])
        stats[v] = dict(seed_means=sm, folds=fl, params=per_seed[seeds[0]][v]['params'])
        coll = int((fl < args.collapse_threshold).sum())
        vs = np.mean([per_seed[s][v]['vsel_acc_mean'] for s in seeds])
        print(f'| {v} | {per_seed[seeds[0]][v]["params"]:,} | '
              f'{sm.mean() * 100:.2f} ± {sm.std() * 100:.2f} | '
              f'{fl.mean() * 100:.2f} ± {fl.std() * 100:.2f} | {fl.min() * 100:.2f} | '
              f'{coll}/{fl.size} ({100 * coll / fl.size:.0f}%) | {vs * 100:.2f} |')

    print('\n## Table 2 — paired comparison against the baseline\n')
    print('| Contrast | ΔACC (pp) | 95% CI | improved/worse/tied | param ratio | '
          'collapse Δ | p (paired t) | p (Wilcoxon) | sign-test p (one-sided) |')
    print('|---|---|---|---|---|---|---|---|---|')
    from scipy import stats as st
    for v in variants:
        if v == BASELINE:
            continue
        a = stats[v]['seed_means']
        d = (a - base) * 100
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
        win, lose = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        tie = n - win - lose
        tp = st.ttest_rel(a, base).pvalue if not np.allclose(a, base) else 1.0
        try:
            wp = st.wilcoxon(a, base).pvalue
        except ValueError:
            wp = float('nan')
        try:
            sp = st.binomtest(win, max(win + lose, 1), 0.5, alternative='greater').pvalue
        except Exception:
            sp = float('nan')
        cv = int((stats[v]['folds'] < args.collapse_threshold).sum())
        cb = int((stats[BASELINE]['folds'] < args.collapse_threshold).sum())
        print(f'| {v} − baseline | {d.mean():+.2f} ± {d.std():.2f} | '
              f'[{d.mean() - 1.96 * se:+.2f}, {d.mean() + 1.96 * se:+.2f}] | '
              f'{win}/{lose}/{tie} | {stats[v]["params"] / stats[BASELINE]["params"]:.4f}× | '
              f'{cv - cb:+d} | {tp:.4f} | {wp:.4f} | {sp:.4f} |')

    main_v = 'AT-DGNN-BandAttn' if 'AT-DGNN-BandAttn' in variants else variants[1]
    d = (stats[main_v]['seed_means'] - base) * 100
    dzv = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    print(f'\n## Power note\n')
    print(f'- main variant `{main_v}`: Cohen $d_z$ = {dzv:.3f}; '
          f'seeds required for 80% power ≈ {int(np.ceil(((2.802 + 0.842) / abs(dzv)) ** 2)) if dzv else float("inf")}')

    if args.show_per_seed:
        print('\n## Table 3 — per-seed 3-fold accuracy (%)\n')
        print('| Model | ' + ' | '.join(f'seed {s}' for s in seeds) + ' |')
        print('|---' * (n + 1) + '|')
        for v in variants:
            cells = ' | '.join('/'.join(f'{x * 100:.1f}' for x in per_seed[s][v]['fold_acc'])
                               for s in seeds)
            print(f'| {v} | {cells} |')

    print('\n## Notes\n')
    print(f'- collapse = fold accuracy below {args.collapse_threshold:.2f} '
          f'(the normal plateau is ~0.95, so a collapse is a failed run)')
    print('- the baseline is taken from archived runs; `band_attn=\'none\'` was verified '
          'bit-identical to AT-DGNN, so the pairing is valid')
    print('- absolute numbers are single-subject pilot numbers and are not comparable '
          'with the 32-subject results of the AT-DGNN paper')


if __name__ == '__main__':
    main()
