"""Check AutoDL environment, dataset layout, RLE, and one model forward pass."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import albumentations
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

from steel_common import build_model, load_annotation_table, rle_decode, rle_encode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    print(f"Python/platform: {platform.python_version()} / {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Albumentations: {albumentations.__version__}")
    print(f"segmentation-models-pytorch: {smp.__version__}")

    table = load_annotation_table(args.data_dir)
    train_count = len(list((args.data_dir / "train_images").glob("*.jpg")))
    test_count = len(list((args.data_dir / "test_images").glob("*.jpg")))
    print(f"annotation table: {table.shape}")
    print(f"images: train={train_count}, test={test_count}")
    print(f"positive masks by class: {table.notna().sum().astype(int).tolist()}")
    if train_count != 12568 or test_count != 5506:
        print("WARNING: image counts differ from the expected competition archive")

    first_nonempty = table.stack().iloc[0]
    mask = rle_decode(first_nonempty)
    roundtrip = np.array_equal(mask, rle_decode(rle_encode(mask)))
    print(f"RLE roundtrip: {roundtrip}; mask pixels={int(mask.sum())}")
    if not roundtrip:
        raise RuntimeError("RLE roundtrip failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("unet", "resnet34", None).to(device).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 400, device=device))
    print(f"model forward output: {tuple(output.shape)}")
    if tuple(output.shape) != (1, 4, 64, 400):
        raise RuntimeError("Unexpected model output shape")
    print("ENVIRONMENT CHECK PASSED")


if __name__ == "__main__":
    main()

