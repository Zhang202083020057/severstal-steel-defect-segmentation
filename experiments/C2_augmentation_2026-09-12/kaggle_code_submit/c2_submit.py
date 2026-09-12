"""Offline inference for the C2-targeted-augmentation experiment."""

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

    checkpoints = list(Path("/kaggle/input").rglob("best_model.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one C2 checkpoint, found: {checkpoints}")

    source_dirs = [
        path.parent
        for path in Path("/kaggle/input").rglob("steel_common.py")
        if (path.parent / "predict.py").is_file()
        and "severstal-c2-training-source" in str(path.parent)
    ]
    if len(source_dirs) != 1:
        raise RuntimeError(f"Expected one C2 source directory, found: {source_dirs}")

    test_dirs = [
        path
        for path in Path("/kaggle/input/competitions").rglob("test_images")
        if path.is_dir()
    ]
    if len(test_dirs) != 1:
        raise RuntimeError(f"Expected one competition test_images directory, found: {test_dirs}")

    work_dir = Path("/kaggle/working")
    shutil.copy2(checkpoints[0], work_dir / "best_model.pt")
    for filename in ("predict.py", "steel_common.py"):
        shutil.copy2(source_dirs[0] / filename, work_dir / filename)

    output = work_dir / "submission.csv"
    subprocess.check_call(
        [
            sys.executable,
            str(work_dir / "predict.py"),
            "--data-dir",
            str(test_dirs[0].parent),
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
        raise RuntimeError("Inference ended without submission.csv")
    line_count = sum(1 for _ in output.open("r", encoding="utf-8")) - 1
    if line_count != 22_024:
        raise RuntimeError(f"Expected 22,024 submission rows, found {line_count}")
    print(f"SUBMISSION_READY: {output} rows={line_count}")


if __name__ == "__main__":
    main()
