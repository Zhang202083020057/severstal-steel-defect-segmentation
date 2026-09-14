"""Generate a competition submission from the completed C2 crop kernel."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


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
        if (p.parent / "predict.py").is_file()
        and "severstal-c2-training-source" in str(p.parent)
    ]
    if len(source_dirs) != 1:
        raise RuntimeError(f"Expected one source directory, found: {source_dirs}")
    checkpoints = [
        p for p in Path("/kaggle/input").rglob("best_model.pt")
        if "severstal-c2-region-crop-train" in str(p)
    ]
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one C2 crop checkpoint, found: {checkpoints}")
    checkpoint = checkpoints[0]
    if checkpoint.stat().st_size < 1_000_000:
        raise RuntimeError(f"Checkpoint appears invalid: {checkpoint} ({checkpoint.stat().st_size} bytes)")
    test_dirs = list(Path("/kaggle/input/competitions").rglob("test_images"))
    if len(test_dirs) != 1:
        raise RuntimeError(f"Expected one test_images directory, found: {test_dirs}")

    work = Path("/kaggle/working/c2_crop_submit_source")
    work.mkdir(parents=True, exist_ok=True)
    for name in ("predict.py", "steel_common.py"):
        shutil.copy2(source_dirs[0] / name, work / name)
    output = Path("/kaggle/working/submission.csv")
    command = [
        sys.executable, "predict.py", "--data-dir", str(test_dirs[0].parent),
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--thresholds", "0.5,0.5,0.5,0.075",
        "--min-total-pixels", "0,0,800,800", "--min-sizes", "0,0,0,0",
        "--tta", "none", "--submission-format", "original",
        "--batch-size", "4", "--num-workers", "2",
    ]
    print("CHECKPOINT:", checkpoint, checkpoint.stat().st_size, "bytes", flush=True)
    print("RUN:", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=work)

    submission = pd.read_csv(output)
    if len(submission) != 22024:
        raise RuntimeError(f"Unexpected row count: {len(submission)}")
    classes = submission["ImageId_ClassId"].str.rsplit("_", n=1).str[-1]
    nonempty = submission["EncodedPixels"].notna()
    print("rows:", len(submission))
    for class_id in ("1", "2", "3", "4"):
        print(f"C{class_id} nonempty:", int((nonempty & classes.eq(class_id)).sum()))


if __name__ == "__main__":
    main()
