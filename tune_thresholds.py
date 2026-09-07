"""Search one probability threshold per defect class on the validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from steel_common import (
    NUM_CLASSES,
    SteelDataset,
    build_model,
    build_transforms,
    dice_sums,
    load_annotation_table,
    save_json,
    split_ids,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    grid = [float(value) for value in args.grid.split(",")]
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = build_model(checkpoint["architecture"], checkpoint["encoder"], None).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])
    table = load_annotation_table(args.data_dir)
    _, val_ids = split_ids(table, args.val_size, args.seed)
    transform = build_transforms(
        checkpoint["height"], checkpoint["width"], training=False, strong=False
    )
    dataset = SteelDataset(args.data_dir / "train_images", val_ids, transform, table)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    sums = torch.zeros(len(grid), NUM_CLASSES, device="cuda")
    counts = torch.zeros(len(grid), NUM_CLASSES, device="cuda")
    for images, masks in tqdm(loader, desc="threshold search"):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        for grid_index, threshold in enumerate(grid):
            batch_sum, batch_count = dice_sums(logits, masks, [threshold] * NUM_CLASSES)
            sums[grid_index] += batch_sum
            counts[grid_index] += batch_count

    scores = (sums / counts.clamp_min(1)).cpu().numpy()
    best_indices = scores.argmax(axis=0)
    best_thresholds = [grid[index] for index in best_indices]
    best_class_dice = [float(scores[index, class_index]) for class_index, index in enumerate(best_indices)]
    result = {
        "thresholds": best_thresholds,
        "class_dice": best_class_dice,
        "mean_dice": float(np.mean(best_class_dice)),
        "grid": grid,
        "score_table": scores.tolist(),
    }
    save_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()

