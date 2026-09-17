#!/bin/bash
# Band-attention sweep with the corrected design (residual fusion + zero-initialised
# score head), paired over the same seeds as the archived baseline.
cd "$(dirname "$0")/.."
PY=/d/SoftWare/Tools/anaconda3/envs/atdgnn/python.exe
VARIANTS=("AT-DGNN-BandAttn" "BandAttn (static)" "BandAttn (adaptive)")
for S in "$@"; do
  OUT="experiments/results/bandattn2_seed${S}.json"
  if [ -f "$OUT" ]; then echo "=== bandattn2 seed $S already done, skipping ==="; continue; fi
  echo "=== bandattn2 seed $S starting $(date) ==="
  "$PY" -u experiments/pilot_bandattn.py --device cuda --folds 3 --epochs 40 --seed "$S" \
     --band-fuse residual --variants "${VARIANTS[@]}" --out "$OUT" \
     > "experiments/results/bandattn2_${S}.log" 2>&1
  echo "=== bandattn2 seed $S done $(date) ==="
done
