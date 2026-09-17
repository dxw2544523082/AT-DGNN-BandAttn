#!/bin/bash
# Multi-scale (frequency-scale) attention sweep, paired over the same seeds.
# Baseline is reused from the archive (verified bit-identical; scale_attn='none' == baseline).
cd "$(dirname "$0")/.."
PY=/d/SoftWare/Tools/anaconda3/envs/atdgnn/python.exe
for S in "$@"; do
  OUT="experiments/results/scaleattn_seed${S}.json"
  if [ -f "$OUT" ]; then echo "=== scaleattn seed $S already done, skipping ==="; continue; fi
  echo "=== scaleattn seed $S starting $(date) ==="
  "$PY" -u experiments/pilot_bandattn.py --device cuda --folds 3 --epochs 40 --seed "$S" \
     --variant-set scale --out "$OUT" > "experiments/results/scaleattn_${S}.log" 2>&1
  echo "=== scaleattn seed $S done $(date) ==="
done
