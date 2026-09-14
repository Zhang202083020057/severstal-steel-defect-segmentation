"""Shared data, model, loss and metric utilities for the Severstal project."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


ORIGINAL_HEIGHT = 256
ORIGINAL_WIDTH = 1600
NUM_CLASSES = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rle_decode(encoded: object, height: int = ORIGINAL_HEIGHT, width: int = ORIGINAL_WIDTH) -> np.ndarray:
    """Decode the competition's 1-indexed, column-major RLE."""
    flat = np.zeros(height * width, dtype=np.uint8)
    if encoded is None or pd.isna(encoded) or not str(encoded).strip():
        return flat.reshape((height, width), order="F")
    values = np.asarray(str(encoded).split(), dtype=np.int64)
    if len(values) % 2:
        raise ValueError("RLE must consist of start/length pairs")
    starts = values[0::2] - 1
    lengths = values[1::2]
    for start, length in zip(starts, lengths):
        end = start + length
        if start < 0 or end > flat.size:
            raise ValueError(f"Invalid RLE interval: start={start + 1}, length={length}")
        flat[start:end] = 1
    return flat.reshape((height, width), order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask with the competition's column-major convention."""
    pixels = (mask > 0).astype(np.uint8).reshape(-1, order="F")
    padded = np.concatenate(([0], pixels, [0]))
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))


def load_annotation_table(data_dir: Path) -> pd.DataFrame:
    """Build a table with one row per image and one RLE column per class."""
    csv_path = data_dir / "train.csv"
    image_dir = data_dir / "train_images"
    if not csv_path.exists() or not image_dir.exists():
        raise FileNotFoundError(f"Missing train.csv or train_images under {data_dir}")
    annotations = pd.read_csv(csv_path)
    expected = {"ImageId", "ClassId", "EncodedPixels"}
    if not expected.issubset(annotations.columns):
        raise ValueError(f"train.csv needs columns {sorted(expected)}")
    table = annotations.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
    table = table.reindex(columns=range(1, NUM_CLASSES + 1))
    image_ids = sorted(path.name for path in image_dir.glob("*.jpg"))
    table = table.reindex(image_ids)
    table.index.name = "ImageId"
    return table


def make_targets(table: pd.DataFrame) -> np.ndarray:
    return table.notna().astype(np.uint8).to_numpy()


def split_ids(table: pd.DataFrame, val_size: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic split stratified by label combination where possible."""
    target = make_targets(table)
    signatures = np.asarray(["".join(map(str, row)) for row in target])
    values, counts = np.unique(signatures, return_counts=True)
    count_map = dict(zip(values, counts))
    largest = values[int(np.argmax(counts))]
    strata = np.asarray([value if count_map[value] >= 2 else largest for value in signatures])
    train_ids, val_ids = train_test_split(
        table.index.to_numpy(), test_size=val_size, random_state=seed, stratify=strata
    )
    return train_ids.tolist(), val_ids.tolist()


def build_transforms(height: int, width: int, training: bool, strong: bool) -> A.Compose:
    transforms: list[A.BasicTransform] = [A.Resize(height, width)]
    if training:
        transforms.append(A.HorizontalFlip(p=0.5))
        if strong:
            transforms.extend(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=0.4
                    ),
                    A.RandomGamma(gamma_limit=(85, 115), p=0.25),
                    A.GaussNoise(p=0.15),
                ]
            )
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ]
    )
    return A.Compose(transforms)


def build_c2_transforms(height: int, width: int) -> A.Compose:
    """Build conservative stronger transforms used only by C2 training samples."""
    noise_parameters = inspect.signature(A.GaussNoise).parameters
    if "std_range" in noise_parameters:
        noise = A.GaussNoise(std_range=(0.01, 0.03), p=0.15)
    else:
        # Albumentations 1.x expresses the same scale as variance in pixel units.
        noise = A.GaussNoise(var_limit=(6.5, 58.5), p=0.15)
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.25),
            noise,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ]
    )


class SteelDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        image_ids: list[str],
        transform: A.Compose,
        table: pd.DataFrame | None = None,
        c2_transform: A.Compose | None = None,
        c2_crop: bool = False,
        c2_crop_width: int = 800,
    ) -> None:
        if c2_transform is not None and table is None:
            raise ValueError("C2 augmentation requires an annotation table")
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transform = transform
        self.table = table
        self.c2_transform = c2_transform
        self.c2_crop = c2_crop
        self.c2_crop_width = c2_crop_width

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(str(self.image_dir / image_id), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_dir / image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.table is None:
            result = self.transform(image=image)
            return result["image"], image_id

        row = self.table.loc[image_id]
        mask = np.stack([rle_decode(row[class_id]) for class_id in range(1, 5)], axis=-1)
        transform = self.transform
        if (
            self.c2_transform is not None
            and pd.notna(row[2])
            and bool(str(row[2]).strip())
        ):
            transform = self.c2_transform
            if self.c2_crop:
                # Crop a random horizontal window that contains the C2 mask.
                # The crop is applied only during training; validation remains
                # on the original image distribution.
                c2 = mask[..., 1]
                ys, xs = np.where(c2 > 0)
                crop_width = min(self.c2_crop_width, image.shape[1])
                if len(xs) and crop_width < image.shape[1]:
                    left_min = max(0, int(xs.max()) - crop_width + 1)
                    left_max = min(int(xs.min()), image.shape[1] - crop_width)
                    if left_min <= left_max:
                        left = int(np.random.randint(left_min, left_max + 1))
                    else:
                        center = int((xs.min() + xs.max()) / 2)
                        left = int(np.clip(center - crop_width // 2, 0, image.shape[1] - crop_width))
                    image = image[:, left:left + crop_width]
                    mask = mask[:, left:left + crop_width]
        result = transform(image=image, mask=mask)
        image_tensor = result["image"].float()
        mask_tensor = result["mask"].float()
        if mask_tensor.shape[0] != NUM_CLASSES and mask_tensor.shape[-1] == NUM_CLASSES:
            mask_tensor = mask_tensor.permute(2, 0, 1)
        return image_tensor, mask_tensor


def build_model(architecture: str, encoder: str, encoder_weights: str | None) -> nn.Module:
    common = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    if architecture == "unet":
        return smp.Unet(**common)
    if architecture == "fpn":
        return smp.FPN(**common)
    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)
    raise ValueError(f"Unknown architecture: {architecture}")


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0
) -> torch.Tensor:
    """Binary focal loss for four independent segmentation channels."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        region_loss = soft_dice_loss(logits, targets)
        if self.name == "bce_dice":
            pixel_loss = F.binary_cross_entropy_with_logits(logits, targets)
        elif self.name == "focal_dice":
            pixel_loss = focal_loss(logits, targets)
        else:
            raise ValueError(f"Unknown loss: {self.name}")
        return pixel_loss + region_loss


def dice_sums(
    logits: torch.Tensor, targets: torch.Tensor, thresholds: Iterable[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-class sums/counts using one Dice per image-class pair."""
    threshold_tensor = torch.as_tensor(
        list(thresholds), device=logits.device, dtype=logits.dtype
    ).view(1, NUM_CLASSES, 1, 1)
    prediction = torch.sigmoid(logits) > threshold_tensor
    truth = targets > 0.5
    intersection = (prediction & truth).sum(dim=(2, 3)).float()
    denominator = prediction.sum(dim=(2, 3)).float() + truth.sum(dim=(2, 3)).float()
    dice = torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        2.0 * intersection / denominator.clamp_min(1.0),
    )
    return dice.sum(dim=0), torch.full(
        (NUM_CLASSES,), dice.shape[0], device=dice.device, dtype=torch.float32
    )


def remove_small_components(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    if minimum_size <= 0 or not mask.any():
        return mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_size:
            cleaned[labels == label] = 1
    return cleaned


def parse_number_list(value: str, cast, expected: int = NUM_CLASSES) -> list:
    result = [cast(item.strip()) for item in value.split(",")]
    if len(result) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {value!r}")
    return result


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
