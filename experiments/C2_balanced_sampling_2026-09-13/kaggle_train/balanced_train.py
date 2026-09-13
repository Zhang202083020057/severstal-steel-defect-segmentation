"""Run the controlled C2 balanced-sampling experiment on Kaggle."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run(*arguments: str, cwd: Path) -> None:
    command = [sys.executable, "-u", *arguments]
    print("RUN:", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=cwd)


def main() -> None:
    wheel_matches = list(
        Path("/kaggle/input").rglob("segmentation_models_pytorch-0.5.0-py3-none-any.whl")
    )
    if len(wheel_matches) != 1:
        raise RuntimeError(f"Expected one offline wheel directory, found: {wheel_matches}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--find-links",
            str(wheel_matches[0].parent),
            "Pillow==11.3.0",
            "timm==1.0.29",
            "segmentation-models-pytorch==0.5.0",
        ]
    )

    source_dirs = [
        path.parent
        for path in Path("/kaggle/input").rglob("steel_common.py")
        if (path.parent / "train.py").is_file()
        and (path.parent / "predict.py").is_file()
        and "severstal-balanced-training-source" in str(path.parent)
    ]
    if len(source_dirs) != 1:
        raise RuntimeError(f"Expected one balanced source directory, found: {source_dirs}")

    train_dirs = [
        path
        for path in Path("/kaggle/input/competitions").rglob("train_images")
        if path.is_dir()
    ]
    if len(train_dirs) != 1:
        raise RuntimeError(f"Expected one competition train_images directory, found: {train_dirs}")

    source_dir = Path("/kaggle/working/balanced_source")
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("steel_common.py", "train.py", "predict.py"):
        shutil.copy2(source_dirs[0] / filename, source_dir / filename)

    output_dir = Path("/kaggle/working/outputs/C2_balanced_unet_resnet34")
    run(
        "train.py",
        "--data-dir", str(train_dirs[0].parent),
        "--output-dir", str(output_dir),
        "--architecture", "unet",
        "--encoder", "resnet34",
        "--encoder-weights", "imagenet",
        "--loss", "bce_dice",
        "--augmentation", "basic",
        "--c2-augmentation",
        "--sampling", "sqrt_inverse_frequency",
        "--height", "256",
        "--width", "800",
        "--epochs", "10",
        "--batch-size", "4",
        "--learning-rate", "0.0003",
        "--num-workers", "2",
        "--seed", "42",
        cwd=source_dir,
    )

    run(
        "predict.py",
        "--data-dir", str(train_dirs[0].parent),
        "--checkpoint", str(output_dir / "best_model.pt"),
        "--output", "/kaggle/working/submission.csv",
        "--thresholds", "0.5,0.5,0.5,0.5",
        "--tta", "none",
        "--submission-format", "original",
        "--batch-size", "4",
        "--num-workers", "2",
        cwd=source_dir,
    )
    print("BALANCED_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
