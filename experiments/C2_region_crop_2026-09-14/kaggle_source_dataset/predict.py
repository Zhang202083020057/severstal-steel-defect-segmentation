"""Run test-time inference, optional horizontal-flip TTA, and write submission CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from steel_common import (
    ORIGINAL_HEIGHT,
    ORIGINAL_WIDTH,
    SteelDataset,
    build_model,
    build_transforms,
    parse_number_list,
    remove_small_components,
    rle_encode,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--thresholds", default=None, help="Example: 0.5,0.5,0.5,0.5")
    parser.add_argument(
        "--min-total-pixels",
        default="0,0,0,0",
        help="Suppress a whole class mask when its total area is below this value",
    )
    parser.add_argument("--min-sizes", default="0,0,0,0", help="Minimum component pixels per class")
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--submission-format", choices=["compact", "original"], default="compact")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["architecture"], checkpoint["encoder"], None).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    thresholds = (
        parse_number_list(args.thresholds, float)
        if args.thresholds
        else checkpoint.get("thresholds", [0.5] * 4)
    )
    minimum_sizes = parse_number_list(args.min_sizes, int)
    minimum_total_pixels = parse_number_list(args.min_total_pixels, int)
    threshold_tensor = torch.tensor(thresholds, device=device).view(1, 4, 1, 1)

    test_dir = args.data_dir / "test_images"
    image_ids = sorted(path.name for path in test_dir.glob("*.jpg"))
    if args.max_samples > 0:
        image_ids = image_ids[: args.max_samples]
    transform = build_transforms(
        checkpoint["height"], checkpoint["width"], training=False, strong=False
    )
    dataset = SteelDataset(test_dir, image_ids, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    for images, batch_ids in tqdm(loader, desc="predict"):
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            probabilities = torch.sigmoid(model(images))
            if args.tta == "hflip":
                flipped_probabilities = torch.sigmoid(
                    model(torch.flip(images, dims=[3]))
                )
                probabilities = (probabilities + torch.flip(flipped_probabilities, dims=[3])) / 2
        probabilities = F.interpolate(
            probabilities,
            size=(ORIGINAL_HEIGHT, ORIGINAL_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        predictions = (probabilities > threshold_tensor).cpu().numpy()

        for batch_index, image_id in enumerate(batch_ids):
            predicted_anything = False
            for class_index in range(4):
                mask = predictions[batch_index, class_index].astype("uint8")
                if int(mask.sum()) < minimum_total_pixels[class_index]:
                    mask.fill(0)
                mask = remove_small_components(mask, minimum_sizes[class_index])
                encoded = rle_encode(mask)
                class_id = class_index + 1
                if args.submission_format == "original":
                    rows.append(
                        {"ImageId_ClassId": f"{image_id}_{class_id}", "EncodedPixels": encoded}
                    )
                elif encoded:
                    predicted_anything = True
                    rows.append(
                        {"ImageId": image_id, "EncodedPixels": encoded, "ClassId": class_id}
                    )
            if args.submission_format == "compact" and not predicted_anything:
                rows.append(
                    {
                        "ImageId": image_id,
                        "EncodedPixels": f"1 {ORIGINAL_HEIGHT * ORIGINAL_WIDTH}",
                        "ClassId": 0,
                    }
                )

    fieldnames = (
        ["ImageId", "EncodedPixels", "ClassId"]
        if args.submission_format == "compact"
        else ["ImageId_ClassId", "EncodedPixels"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"thresholds={thresholds}, min_total_pixels={minimum_total_pixels}, "
        f"min_sizes={minimum_sizes}, TTA={args.tta}"
    )
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
