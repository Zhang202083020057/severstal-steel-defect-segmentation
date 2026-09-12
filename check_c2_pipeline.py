"""Audit the B0 split and save C2 online-augmentation previews without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from steel_common import (
    SteelDataset,
    build_c2_transforms,
    build_transforms,
    load_annotation_table,
    make_targets,
    rle_decode,
    save_json,
    seed_everything,
    split_ids,
)


B0_TOTAL_IMAGES = 12_568
B0_TOTAL_CLASS_COUNTS = [897, 247, 5_150, 801]
B0_TRAIN_IMAGES = 10_054
B0_TRAIN_CLASS_COUNTS = [718, 198, 4_120, 641]
B0_VAL_IMAGES = 2_514
B0_VAL_CLASS_COUNTS = [179, 49, 1_030, 160]
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
C2_COLOR = np.asarray([80, 220, 80], dtype=np.float32)


class CountingTransform:
    """Count calls while forwarding to a real Albumentations transform."""

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self.transform(**kwargs)


def class_counts(table, image_ids: list[str]) -> list[int]:
    return make_targets(table.loc[image_ids]).sum(axis=0).astype(int).tolist()


def transform_names(transform: A.Compose) -> list[str]:
    return [type(item).__name__ for item in transform.transforms]


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image * IMAGE_STD + IMAGE_MEAN) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def overlay_c2(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    foreground = mask.astype(bool)
    overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * C2_COLOR
    return np.clip(overlay, 0, 255).astype(np.uint8)


def verify_synthetic_image_mask_sync() -> float:
    """Force geometry changes and verify that image and mask stay aligned."""
    mask = np.zeros((96, 240), dtype=np.uint8)
    mask[25:70, 45:130] = 1
    image = np.zeros((96, 240, 3), dtype=np.uint8)
    image[mask == 1, 0] = 255
    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(5, 5), border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToTensorV2(transpose_mask=True),
        ]
    )
    result = transform(image=image, mask=mask)
    image_foreground = result["image"][0].numpy() >= 128
    mask_foreground = result["mask"].numpy() > 0
    intersection = np.logical_and(image_foreground, mask_foreground).sum()
    union = np.logical_or(image_foreground, mask_foreground).sum()
    overlap = intersection / max(union, 1)
    if overlap < 0.98:
        raise AssertionError(f"Image/mask geometric alignment check failed: IoU={overlap:.4f}")
    return float(overlap)


def verify_b0_counts(
    total_images: int,
    total_counts: list[int],
    train_ids: list[str],
    train_counts: list[int],
    val_ids: list[str],
    val_counts: list[int],
) -> None:
    actual = {
        "total_images": total_images,
        "total_counts": total_counts,
        "train_images": len(train_ids),
        "train_counts": train_counts,
        "val_images": len(val_ids),
        "val_counts": val_counts,
    }
    expected = {
        "total_images": B0_TOTAL_IMAGES,
        "total_counts": B0_TOTAL_CLASS_COUNTS,
        "train_images": B0_TRAIN_IMAGES,
        "train_counts": B0_TRAIN_CLASS_COUNTS,
        "val_images": B0_VAL_IMAGES,
        "val_counts": B0_VAL_CLASS_COUNTS,
    }
    if actual != expected:
        raise AssertionError(f"The data split does not match B0. expected={expected}, actual={actual}")


def save_previews(
    data_dir: Path,
    output_dir: Path,
    table,
    train_ids: list[str],
    c2_ids: list[str],
    base_transform: A.Compose,
    c2_transform: A.Compose,
    count: int,
    variants: int,
    seed: int,
) -> list[str]:
    selected_count = min(count, len(c2_ids))
    generator = np.random.default_rng(seed)
    selected_ids = generator.choice(c2_ids, size=selected_count, replace=False).tolist()
    if hasattr(c2_transform, "set_random_seed"):
        c2_transform.set_random_seed(seed)
    dataset = SteelDataset(
        data_dir / "train_images",
        train_ids,
        base_transform,
        table,
        c2_transform=c2_transform,
    )
    positions = {image_id: index for index, image_id in enumerate(train_ids)}
    saved_paths = []

    for image_id in selected_ids:
        image = cv2.imread(str(data_dir / "train_images" / image_id), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(data_dir / "train_images" / image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_mask = rle_decode(table.loc[image_id, 2])

        figure, axes = plt.subplots(
            1, variants + 1, figsize=(5 * (variants + 1), 3.2), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        axes[0].imshow(overlay_c2(image, original_mask))
        axes[0].set_title(f"Original + C2 mask\n{image_id}")
        axes[0].axis("off")

        for variant in range(variants):
            image_tensor, mask_tensor = dataset[positions[image_id]]
            augmented_image = denormalize_image(image_tensor)
            augmented_mask = mask_tensor[1].detach().cpu().numpy() > 0.5
            axes[variant + 1].imshow(overlay_c2(augmented_image, augmented_mask))
            axes[variant + 1].set_title(f"Online augmentation {variant + 1}")
            axes[variant + 1].axis("off")

        output_path = output_dir / f"{Path(image_id).stem}_c2_augmentation.png"
        figure.savefig(output_path, dpi=130)
        plt.close(figure)
        saved_paths.append(str(output_path))

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the unchanged B0 split and preview C2-only online augmentation."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c2_augmentation_check")
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-b0-count-check",
        action="store_true",
        help="Allow a small or nonstandard dataset; intended only for smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.variants < 1:
        raise ValueError("--count and --variants must both be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    table = load_annotation_table(args.data_dir)
    train_ids, val_ids = split_ids(table, args.val_size, args.seed)
    if set(train_ids) & set(val_ids):
        raise AssertionError("Data leakage detected: train and validation IDs overlap")
    if set(train_ids) | set(val_ids) != set(table.index):
        raise AssertionError("Split does not cover every original image exactly once")

    total_counts = make_targets(table).sum(axis=0).astype(int).tolist()
    train_counts = class_counts(table, train_ids)
    val_counts = class_counts(table, val_ids)
    if not args.skip_b0_count_check:
        verify_b0_counts(
            len(table), total_counts, train_ids, train_counts, val_ids, val_counts
        )

    train_targets = make_targets(table.loc[train_ids])
    c2_ids = [
        image_id for image_id, target in zip(train_ids, train_targets) if target[1] == 1
    ]
    non_c2_ids = [
        image_id for image_id, target in zip(train_ids, train_targets) if target[1] == 0
    ]
    if not c2_ids or not non_c2_ids:
        raise AssertionError("Both C2 and non-C2 training samples are required for this check")

    base_transform = build_transforms(
        args.height, args.width, training=True, strong=False
    )
    c2_transform = build_c2_transforms(args.height, args.width)
    valid_transform = build_transforms(
        args.height, args.width, training=False, strong=False
    )
    expected_validation = ["Resize", "Normalize", "ToTensorV2"]
    if transform_names(valid_transform) != expected_validation:
        raise AssertionError(
            f"Validation contains unexpected transforms: {transform_names(valid_transform)}"
        )

    base_counter = CountingTransform(base_transform)
    c2_counter = CountingTransform(c2_transform)
    routing_dataset = SteelDataset(
        args.data_dir / "train_images",
        [non_c2_ids[0], c2_ids[0]],
        base_counter,
        table,
        c2_transform=c2_counter,
    )
    routing_dataset[0]
    if (base_counter.calls, c2_counter.calls) != (1, 0):
        raise AssertionError("A non-C2 sample did not use basic augmentation exclusively")
    routing_dataset[1]
    if (base_counter.calls, c2_counter.calls) != (1, 1):
        raise AssertionError("A C2 sample did not use C2 augmentation exclusively")
    image_mask_sync_iou = verify_synthetic_image_mask_sync()

    preview_paths = save_previews(
        args.data_dir,
        args.output_dir,
        table,
        train_ids,
        c2_ids,
        base_transform,
        c2_transform,
        args.count,
        args.variants,
        args.seed,
    )
    report = {
        "status": "passed",
        "note": "Preview PNGs are audit artifacts and are never added to train_images.",
        "split": {
            "seed": args.seed,
            "val_size": args.val_size,
            "total_images": len(table),
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "total_class_counts": total_counts,
            "train_class_counts": train_counts,
            "val_class_counts": val_counts,
            "train_val_overlap": 0,
        },
        "augmentation": {
            "non_c2": transform_names(base_transform),
            "c2": transform_names(c2_transform),
            "c2_parameters": {
                "horizontal_flip_probability": 0.5,
                "rotation_limit_degrees": 2,
                "rotation_probability": 0.2,
                "brightness_limit": 0.15,
                "contrast_limit": 0.15,
                "brightness_contrast_probability": 0.4,
                "gamma_limit": [85, 115],
                "gamma_probability": 0.25,
                "gaussian_noise_std_range": [0.01, 0.03],
                "gaussian_noise_probability": 0.15,
            },
            "validation": transform_names(valid_transform),
            "oversampling": False,
            "image_mask_sync": {
                "status": "passed",
                "synthetic_geometric_iou": image_mask_sync_iou,
                "minimum_required_iou": 0.98,
            },
            "routing": "passed",
        },
        "preview_files": preview_paths,
    }
    report_path = args.output_dir / "c2_pipeline_report.json"
    save_json(report_path, report)

    print("C2 data-pipeline audit passed")
    print(f"total={len(table)} train={len(train_ids)} val={len(val_ids)}")
    print(f"class counts total={total_counts} train={train_counts} val={val_counts}")
    print(f"saved {len(preview_paths)} previews and {report_path}")


if __name__ == "__main__":
    main()
