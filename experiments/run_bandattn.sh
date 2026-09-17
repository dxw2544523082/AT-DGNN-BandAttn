#!/bin/bash
# Band-attention sweep, paired over the same seeds as the archived baseline.
# The baseline itself is NOT re-run: band_attn='none' was verified bit-identical
# to the archived AT-DGNN runs (max abs diff 0.0), so the archived JSONs pair directly.
cd "$(dirname "$0")/.."
PY=/d/SoftWare/Tools/anaconda3/envs/atdgnn/python.exe
VARIANTS=("AT-DGNN-BandAttn" "BandAttn (static)" "BandAttn (adaptive)")
for S in "$@"; do
  OUT="experiments/results/bandattn_seed${S}.json"
  if [ -f "$OUT" ]; then echo "=== bandattn seed $S already done, skipping ==="; continue; fi
  echo "=== bandattn seed $S starting $(date) ==="
  "$PY" -u experiments/pilot_attgraph.py --device cuda --folds 3 --epochs 40 --seed "$S" \
     --variants "${VARIANTS[@]}" --out "$OUT" > "experiments/results/bandattn_${S}.log" 2>&1
  echo "=== bandattn seed $S done $(date) ==="
done
