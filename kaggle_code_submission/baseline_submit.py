"""Offline, reproducible B0 inference script for Kaggle code submission.

Required Kaggle inputs:
1. Competition: severstal-steel-defect-detection
2. Private assets dataset: best_model.pt, predict.py, steel_common.py
3. Private wheels dataset: Pillow 11.3.0, timm 1.0.29,
   segmentation-models-pytorch 0.5.0
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def pip_install_offline(wheel_dir: Path, *packages: str) -> None:
    """Install only from an attached Kaggle input; no Internet is required."""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--find-links",
            str(wheel_dir),
            *packages,
        ]
    )


def main() -> None:
    wheel_matches = list(
        Path("/kaggle/input").rglob("segmentation_models_pytorch-0.5.0-py3-none-any.whl")
    )
    if len(wheel_matches) != 1:
        raise RuntimeError(f"Expected one offline wheel input, found: {wheel_matches}")
    pip_install_offline(
        wheel_matches[0].parent,
        "Pillow==11.3.0",
        "timm==1.0.29",
        "segmentation-models-pytorch==0.5.0",
    )

    asset_matches = list(Path("/kaggle/input").rglob("best_model.pt"))
    if len(asset_matches) != 1:
        raise RuntimeError(f"Expected one best_model.pt input, found: {asset_matches}")
    asset_dir = asset_matches[0].parent

    test_dirs = [
        directory
        for directory in Path("/kaggle/input/competitions").rglob("test_images")
        if directory.is_dir()
    ]
    if len(test_dirs) != 1:
        raise RuntimeError(f"Expected one competition test_images directory, found: {test_dirs}")
    data_dir = test_dirs[0].parent

    work_dir = Path("/kaggle/working")
    for filename in ("best_model.pt", "predict.py", "steel_common.py"):
        source = asset_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing required asset: {source}")
        shutil.copy2(source, work_dir / filename)

    output = work_dir / "submission.csv"
    subprocess.check_call(
        [
            sys.executable,
            str(work_dir / "predict.py"),
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(work_dir / "best_model.pt"),
            "--output",
            str(output),
            "--thresholds",
            "0.5,0.5,0.5,0.5",
            "--tta",
            "none",
            "--submission-format",
            "original",
            "--batch-size",
            "4",
            "--num-workers",
            "2",
        ],
        cwd=work_dir,
    )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Inference ended without producing submission.csv")
    print(f"SUBMISSION_READY: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
