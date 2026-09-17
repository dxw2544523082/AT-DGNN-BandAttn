"""
Summarise LOSO runs: paired per-subject comparison against the baseline.

One LOSO run evaluates every variant on exactly the same subjects, so the comparison
is paired by subject with n = #subjects - the inferential unit that a cross-subject
claim actually needs (as opposed to re-using one subject's data across random seeds).

Usage
-----
    python experiments/summarize_loso.py --pattern 'experiments/results/loso_MEEG.json'
"""

import argparse
import glob
import json
import sys

import numpy as np

BASELINE = 'AT-DGNN'


def load(pattern):
    runs = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
        for r in payload['results']:
            runs[r['variant']] = r
        protocol = payload['protocol']
    return runs, protocol


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='experiments/results/loso_MEEG.json')
    ap.add_argument('--collapse-threshold', type=float, default=0.6)
    args = ap.parse_args()

    runs, protocol = load(args.pattern)
    if not runs:
        print(f'no results matched {args.pattern}')
        return
    names = [BASELINE] + [n for n in runs if n != BASELINE] if BASELINE in runs else list(runs)
    n_sub = len(runs[names[0]]['fold_acc'])

    print('# LOSO (cross-subject) summary\n')
    print('| protocol | value |')
    print('|---|---|')
    for k, v in protocol.items():
        print(f'| {k} | {v} |')

    print(f'\n## Table 1 — accuracy over subjects (n = {n_sub})\n')
    print('| Model | Params | ACC over subjects (%) | ACC over segments (%) | '
          'worst subject (%) | best subject (%) | collapses | time (s) |')
    print('|---|---|---|---|---|---|---|---|')
    for n in names:
        r = runs[n]
        a = np.array(r['fold_acc'])
        n_col = int((a < args.collapse_threshold).sum())
        print(f'| {n} | {r["params"]:,} | {a.mean() * 100:.2f} ± {a.std() * 100:.2f} | '
              f'{r["f1_mean"] * 100:.2f} ± {r["f1_std"] * 100:.2f} | {a.min() * 100:.2f} | '
              f'{a.max() * 100:.2f} | {n_col}/{n_sub} | {r["seconds"]:.0f} |')

    if BASELINE in runs and len(names) > 1:
        b = np.array(runs[BASELINE]['fold_acc'])
        print('\n## Table 2 — paired comparison against the baseline (paired by subject)\n')
        print('| Contrast | ΔACC (pp) | 95% CI | improved / worse / tied | collapsed subjects Δ | p (paired t) | p (Wilcoxon) |')
        print('|---|---|---|---|---|---|---|')
        from scipy import stats as st
        for n in names:
            if n == BASELINE:
                continue
            a = np.array(runs[n]['fold_acc'])
            d = (a - b) * 100
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
            ci = (d.mean() - 1.96 * se, d.mean() + 1.96 * se)
            win, lose = int((d > 1e-9).sum()), int((d < -1e-9).sum())
            cb = int((b < args.collapse_threshold).sum())
            ca = int((a < args.collapse_threshold).sum())
            if np.allclose(a, b):
                tp = wp = 1.0
            else:
                tp = st.ttest_rel(a, b).pvalue
                try:
                    wp = st.wilcoxon(a, b).pvalue
                except ValueError:
                    wp = float('nan')
            print(f'| {n} − baseline | {d.mean():+.2f} ± {d.std():.2f} | '
                  f'[{ci[0]:+.2f}, {ci[1]:+.2f}] | {win}/{lose}/{len(d) - win - lose} | '
                  f'{ca - cb:+d} | {tp:.4f} | {wp:.4f} |')

        print(f'\n## Table 3 — per-subject accuracy (%)\n')
        header = '| Model | ' + ' | '.join(f's{i}' for i in range(n_sub)) + ' |'
        print(header)
        print('|---' * (n_sub + 1) + '|')
        for n in names:
            cells = ' | '.join(f'{x * 100:.1f}' for x in runs[n]['fold_acc'])
            print(f'| {n} | {cells} |')

        # how many subjects does the best variant win on?
        if len(names) > 2:
            print('\n## Table 4 — win counts between the two best variants\n')
            best = sorted([n for n in names if n != BASELINE],
                          key=lambda n: -np.mean(runs[n]['fold_acc']))[:2]
            if len(best) == 2:
                x = np.array(runs[best[0]]['fold_acc'])
                y = np.array(runs[best[1]]['fold_acc'])
                print(f'- {best[0]} beats {best[1]} on {int((x > y).sum())}/{n_sub} subjects '
                      f'(mean Δ {(x - y).mean() * 100:+.2f} pp)')

    print('\n## Notes\n')
    print(f'- collapse = subject accuracy below {args.collapse_threshold:.2f}')
    print('- every variant is evaluated on exactly the same held-out subjects, so the '
          'per-subject differences are paired and the number of independent units is the '
          'number of subjects, not the number of seeds.')


if __name__ == '__main__':
    main()
