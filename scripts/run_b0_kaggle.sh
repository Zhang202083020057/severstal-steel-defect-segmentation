#!/usr/bin/env bash
# Reproduce the B0 baseline on a Kaggle P100 notebook.
# Usage: bash scripts/run_b0_kaggle.sh [DATA_DIR] [OUTPUT_DIR]

set -euo pipefail

DATA_DIR="${1:-/kaggle/input/competitions/severstal-steel-defect-detection}"
OUTPUT_DIR="${2:-/kaggle/working/outputs/B0_unet_resnet34}"

python train.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --architecture unet \
  --encoder resnet34 \
  --encoder-weights imagenet \
  --loss bce_dice \
  --augmentation basic \
  --height 256 \
  --width 800 \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 0.0003 \
  --num-workers 2

# Kaggle notebook submissions require the final file to be named submission.csv.
python predict.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$OUTPUT_DIR/best_model.pt" \
  --output /kaggle/working/submission.csv \
  --thresholds 0.5,0.5,0.5,0.5 \
  --tta none \
  --submission-format original \
  --batch-size 4 \
  --num-workers 2
