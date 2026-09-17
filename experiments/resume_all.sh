#!/bin/bash
# One-command resume of the band-attention experiments after an interruption.
#
#   stage 1: finish the remaining v1 seeds (old 'replace' fusion, kept as the ablation)
#   stage 2: swap in the corrected module, then run the v2 sweep (residual + zero-init)
#
# Both sweeps skip seeds whose result file already exists, so this script is safe to
# re-run at any time and never repeats finished work.
cd "$(dirname "$0")/.."
SEEDS=(3407 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

echo "=== $(date) stage 1: v1 (replace fusion) ==="
bash experiments/run_bandattn.sh "${SEEDS[@]}"
bash experiments/run_bandattn.sh "${SEEDS[@]}"     # second pass catches anything skipped mid-run

echo "=== $(date) stage 2: swap in the corrected module ==="
while [ -n "$(ps -W 2>/dev/null | grep -i atdgnn/python)" ]; do sleep 10; done
# the old pilot imported the discarded variants and is dead code after the swap
rm -f experiments/pilot_attgraph.py experiments/run_bandattn.sh
B=../AT-DGNN_extras/clean_build
cp "$B/networks.clean.py"     models/networks.py
cp "$B/config.clean.py"       config/config.py
cp "$B/utils.clean.py"        utils/utils.py
cp "$B/train_model.clean.py"  train/train_model.py
echo "    clean build installed; v2 uses residual fusion + zero-initialised score head"

echo "=== $(date) stage 3: v2 (residual fusion) ==="
bash experiments/run_bandattn_v2.sh "${SEEDS[@]}"
bash experiments/run_bandattn_v2.sh "${SEEDS[@]}"

echo "=== $(date) all sweeps finished ==="
