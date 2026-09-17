#!/bin/bash
# Cross-subject (LOSO) evaluation for the band-attention variants.
#
# Requires one preprocessed hdf per subject in --data-folder (sub0.hdf ... subN.hdf),
# produced by train/prepare_data.py (see docs/run.md).
#
# Cost guidance (MEEG, 31 training subjects = 8680 segments per epoch, 40 epochs,
# 32 folds): roughly 16 h per variant in fp32 on an RTX 4060 Laptop, ~14 h with AMP.
cd "$(dirname "$0")/.."
PY=python
OUT=${OUT:-experiments/results/loso_MEEG.json}
FOLDER=${FOLDER:-data_eeg_MEEG_A}
"$PY" -u experiments/loso.py \
  --data-folder "$FOLDER" \
  --variants "AT-DGNN" "AT-DGNN-BandAttn" "BandAttn (static)" "BandAttn (adaptive)" \
  --epochs 40 --batch-size 64 --val-subjects 2 --normalize per_subject \
  --out "$OUT"
echo
echo "summarise with:"
echo "  python experiments/summarize_loso.py --pattern '$OUT'"
