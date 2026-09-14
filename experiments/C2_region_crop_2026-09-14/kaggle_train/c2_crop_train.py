"""Train the C2 region-crop experiment on Kaggle."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    wheel_matches = list(
        Path("/kaggle/input").rglob("segmentation_models_pytorch-0.5.0-py3-none-any.whl")
    )
    if len(wheel_matches) != 1:
        raise RuntimeError(f"Expected one offline SMP wheel, found: {wheel_matches}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", str(wheel_matches[0].parent), "Pillow==11.3.0",
        "timm==1.0.29", "segmentation-models-pytorch==0.5.0",
    ])
    source_dirs = [
        p.parent for p in Path("/kaggle/input").rglob("steel_common.py")
        if (p.parent / "train.py").is_file() and (p.parent / "predict.py").is_file()
    ]
    if len(source_dirs) != 1:
        raise RuntimeError(f"Expected one code source directory, found: {source_dirs}")
    train_dirs = list(Path("/kaggle/input/competitions").rglob("train_images"))
    if len(train_dirs) != 1:
        raise RuntimeError(f"Expected one train_images directory, found: {train_dirs}")
    source = source_dirs[0]
    work = Path("/kaggle/working/c2_crop_source")
    work.mkdir(parents=True, exist_ok=True)
    for name in ("steel_common.py", "train.py", "predict.py"):
        shutil.copy2(source / name, work / name)
    data_dir = train_dirs[0].parent
    output = Path("/kaggle/working/outputs/C2_region_crop_unet_resnet34")
    command = [
        sys.executable, "train.py", "--data-dir", str(data_dir),
        "--output-dir", str(output), "--architecture", "unet",
        "--encoder", "resnet34", "--encoder-weights", "imagenet",
        "--loss", "bce_dice", "--augmentation", "basic",
        "--c2-augmentation", "--c2-crop", "--c2-crop-width", "800",
        "--height", "256", "--width", "800", "--epochs", "10",
        "--batch-size", "4", "--learning-rate", "0.0003",
        "--num-workers", "2", "--seed", "42",
    ]
    print("RUN:", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=work)


if __name__ == "__main__":
    main()
