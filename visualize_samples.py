"""Save sample images with colored ground-truth masks for visual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from steel_common import load_annotation_table, rle_decode


COLORS = np.asarray(
    [
        [255, 70, 70],
        [80, 220, 80],
        [80, 140, 255],
        [255, 210, 60],
    ],
    dtype=np.float32,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/samples"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = load_annotation_table(args.data_dir)
    positive_table = table[table.notna().any(axis=1)]
    selected = positive_table.sample(min(args.count, len(positive_table)), random_state=args.seed)
    for image_id, row in selected.iterrows():
        image = cv2.imread(str(args.data_dir / "train_images" / image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        overlay = image.copy()
        present = []
        for class_index in range(4):
            mask = rle_decode(row[class_index + 1]).astype(bool)
            if mask.any():
                present.append(str(class_index + 1))
                overlay[mask] = 0.55 * overlay[mask] + 0.45 * COLORS[class_index]
        figure, axes = plt.subplots(2, 1, figsize=(16, 5), constrained_layout=True)
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title(f"Original: {image_id}")
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title(f"Mask overlay; classes={','.join(present)}")
        for axis in axes:
            axis.axis("off")
        figure.savefig(args.output_dir / f"{Path(image_id).stem}_overlay.png", dpi=130)
        plt.close(figure)
    print(f"saved {len(selected)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
