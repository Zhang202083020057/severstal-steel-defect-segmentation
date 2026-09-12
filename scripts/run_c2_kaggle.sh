#!/usr/bin/env bash
# V2: compare C2-only online augmentation against the V1/B0 baseline on Kaggle.
# All B0 settings and the seed are unchanged; only --c2-augmentation is added.
# Usage: bash scripts/run_c2_kaggle.sh [DATA_DIR] [OUTPUT_DIR]

set -euo pipefail

DATA_DIR="${1:-/kaggle/input/competitions/severstal-steel-defect-detection}"
OUTPUT_DIR="${2:-/kaggle/working/outputs/C2_unet_resnet34}"

python train.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --architecture unet \
  --encoder resnet34 \
  --encoder-weights imagenet \
  --loss bce_dice \
  --augmentation basic \
  --c2-augmentation \
  --height 256 \
  --width 800 \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 0.0003 \
  --num-workers 2 \
  --seed 42

python predict.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$OUTPUT_DIR/best_model.pt" \
  --output /kaggle/working/submission.csv \
  --thresholds 0.5,0.5,0.5,0.5 \
  --tta none \
  --submission-format original \
  --batch-size 4 \
  --num-workers 2
