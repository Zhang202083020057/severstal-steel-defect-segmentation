"""Run V2 probability diagnostics and threshold calibration on Kaggle."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MINIMUM_AREAS = [0, 5, 10, 25, 50, 100, 200, 400]


def install_dependencies() -> None:
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


def percentile_summary(np, values):
    if values.size == 0:
        return {"count": 0}
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "count": int(values.size),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "max": float(quantiles[5]),
    }


def select_per_class(np, scores):
    selected = []
    for class_index in range(4):
        flat_index = int(np.nanargmax(scores[:, :, class_index]))
        threshold_index, area_index = np.unravel_index(
            flat_index, scores[:, :, class_index].shape
        )
        selected.append(
            {
                "class_id": class_index + 1,
                "threshold": THRESHOLDS[threshold_index],
                "minimum_total_pixels_model_space": MINIMUM_AREAS[area_index],
                "score": float(scores[threshold_index, area_index, class_index]),
            }
        )
    return selected


def main() -> None:
    install_dependencies()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    torch.set_grad_enabled(False)

    source_dirs = [
        path.parent
        for path in Path("/kaggle/input").rglob("steel_common.py")
        if "severstal-c2-training-source" in str(path)
        and (path.parent / "train.py").is_file()
    ]
    if len(source_dirs) != 1:
        raise RuntimeError(f"Expected one V2 source directory, found: {source_dirs}")
    sys.path.insert(0, str(source_dirs[0]))

    from steel_common import (
        SteelDataset,
        build_model,
        build_transforms,
        load_annotation_table,
        split_ids,
    )

    checkpoints = list(Path("/kaggle/input").rglob("best_model.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one V2 checkpoint, found: {checkpoints}")
    train_dirs = [
        path
        for path in Path("/kaggle/input/competitions").rglob("train_images")
        if path.is_dir()
    ]
    if len(train_dirs) != 1:
        raise RuntimeError(f"Expected one competition train directory, found: {train_dirs}")

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoints[0], map_location=device, weights_only=False)
    model = build_model(checkpoint["architecture"], checkpoint["encoder"], None).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_dir = train_dirs[0].parent
    table = load_annotation_table(data_dir)
    _, val_ids = split_ids(table, 0.2, 42)
    transform = build_transforms(
        checkpoint["height"], checkpoint["width"], training=False, strong=False
    )
    dataset = SteelDataset(data_dir / "train_images", val_ids, transform, table)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    shape = (len(THRESHOLDS), len(MINIMUM_AREAS), 4)
    dice_sum = torch.zeros(shape, dtype=torch.float64, device=device)
    positive_dice_sum = torch.zeros_like(dice_sum)
    detection_true_positive = torch.zeros_like(dice_sum)
    predicted_nonempty_count = torch.zeros_like(dice_sum)
    total_images = torch.zeros(4, dtype=torch.float64, device=device)
    positive_images = torch.zeros(4, dtype=torch.float64, device=device)
    image_maxima = []
    truth_region_maxima = []
    truth_nonempty_rows = []

    for images, masks in tqdm(loader, desc="threshold calibration"):
        images = images.to(device, non_blocking=True)
        truth = masks.to(device, non_blocking=True) > 0.5
        with torch.autocast("cuda", dtype=torch.float16):
            probabilities = torch.sigmoid(model(images))

        truth_count = truth.sum(dim=(2, 3)).float()
        truth_nonempty = truth_count > 0
        total_images += truth.shape[0]
        positive_images += truth_nonempty.sum(dim=0)
        image_maxima.append(probabilities.amax(dim=(2, 3)).float().cpu())
        truth_region_maxima.append(
            probabilities.masked_fill(~truth, -1.0).amax(dim=(2, 3)).float().cpu()
        )
        truth_nonempty_rows.append(truth_nonempty.cpu())

        for threshold_index, threshold in enumerate(THRESHOLDS):
            prediction = probabilities > threshold
            predicted_area = prediction.sum(dim=(2, 3)).float()
            raw_intersection = (prediction & truth).sum(dim=(2, 3)).float()
            for area_index, minimum_area in enumerate(MINIMUM_AREAS):
                keep = predicted_area >= minimum_area
                predicted_nonempty = keep & (predicted_area > 0)
                kept_area = predicted_area * keep
                intersection = raw_intersection * keep
                denominator = kept_area + truth_count
                dice = torch.where(
                    denominator == 0,
                    torch.ones_like(denominator),
                    2.0 * intersection / denominator.clamp_min(1.0),
                )
                dice_sum[threshold_index, area_index] += dice.sum(dim=0)
                positive_dice_sum[threshold_index, area_index] += (
                    dice * truth_nonempty
                ).sum(dim=0)
                detection_true_positive[threshold_index, area_index] += (
                    predicted_nonempty & truth_nonempty
                ).sum(dim=0)
                predicted_nonempty_count[threshold_index, area_index] += (
                    predicted_nonempty.sum(dim=0)
                )

    overall_dice = dice_sum / total_images.clamp_min(1.0)
    positive_dice = positive_dice_sum / positive_images.clamp_min(1.0)
    detection_recall = detection_true_positive / positive_images.clamp_min(1.0)
    detection_precision = detection_true_positive / predicted_nonempty_count.clamp_min(1.0)
    detection_f1 = (
        2.0 * detection_precision * detection_recall
        / (detection_precision + detection_recall).clamp_min(1e-12)
    )
    predicted_nonempty_rate = predicted_nonempty_count / total_images.clamp_min(1.0)

    overall_np = overall_dice.cpu().numpy()
    positive_np = positive_dice.cpu().numpy()
    detection_f1_np = detection_f1.cpu().numpy()
    maxima = torch.cat(image_maxima).numpy()
    truth_maxima = torch.cat(truth_region_maxima).numpy()
    nonempty = torch.cat(truth_nonempty_rows).numpy().astype(bool)
    probability_summary = []
    for class_index in range(4):
        class_positive = nonempty[:, class_index]
        probability_summary.append(
            {
                "class_id": class_index + 1,
                "positive_image_global_max": percentile_summary(
                    np, maxima[class_positive, class_index]
                ),
                "positive_image_truth_region_max": percentile_summary(
                    np, truth_maxima[class_positive, class_index]
                ),
                "negative_image_global_max": percentile_summary(
                    np, maxima[~class_positive, class_index]
                ),
            }
        )

    result = {
        "checkpoint": str(checkpoints[0]),
        "validation_images": len(val_ids),
        "positive_images_per_class": [int(value) for value in positive_images.cpu().tolist()],
        "model_resolution": [int(checkpoint["height"]), int(checkpoint["width"])],
        "threshold_grid": THRESHOLDS,
        "minimum_area_grid_model_space": MINIMUM_AREAS,
        "best_overall_dice": select_per_class(np, overall_np),
        "best_positive_dice": select_per_class(np, positive_np),
        "best_detection_f1": select_per_class(np, detection_f1_np),
        "probability_summary": probability_summary,
        "overall_dice": overall_np.tolist(),
        "positive_dice": positive_np.tolist(),
        "detection_recall": detection_recall.cpu().numpy().tolist(),
        "detection_precision": detection_precision.cpu().numpy().tolist(),
        "detection_f1": detection_f1_np.tolist(),
        "predicted_nonempty_rate": predicted_nonempty_rate.cpu().numpy().tolist(),
        "note": "Minimum areas are at 256x800 model resolution; test inference upsamples width to 1600.",
    }
    output = Path("/kaggle/working/threshold_results.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best_overall_dice=", result["best_overall_dice"])
    print("best_positive_dice=", result["best_positive_dice"])
    print("best_detection_f1=", result["best_detection_f1"])
    print("probability_summary=", probability_summary)
    print(f"wrote {output}")
    print("THRESHOLD_CALIBRATION_COMPLETE")


if __name__ == "__main__":
    main()
