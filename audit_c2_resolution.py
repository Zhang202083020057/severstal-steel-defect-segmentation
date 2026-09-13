"""Audit how 256x1600 -> 256x800 nearest-neighbor resizing affects Class 2 masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


ORIGINAL_HEIGHT = 256
ORIGINAL_WIDTH = 1600
MODEL_HEIGHT = 256
MODEL_WIDTH = 800


def rle_decode(encoded: str) -> np.ndarray:
    flat = np.zeros(ORIGINAL_HEIGHT * ORIGINAL_WIDTH, dtype=np.uint8)
    values = np.fromstring(encoded, dtype=np.int64, sep=" ")
    if values.size % 2:
        raise ValueError("RLE must contain start/length pairs")
    starts = values[0::2] - 1
    lengths = values[1::2]
    for start, length in zip(starts, lengths):
        flat[start : start + length] = 1
    return flat.reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH), order="F")


def resize_mask_like_opencv_nearest(mask: np.ndarray) -> np.ndarray:
    """Exact mapping for this specific 1600 -> 800 width-only nearest resize."""
    if mask.shape != (ORIGINAL_HEIGHT, ORIGINAL_WIDTH):
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    resized = mask[:, ::2]
    if resized.shape != (MODEL_HEIGHT, MODEL_WIDTH):
        raise AssertionError(resized.shape)
    return resized


def component_areas(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.asarray([], dtype=np.int64)
    return np.bincount(labels.ravel())[1:]


def bounding_box(mask: np.ndarray) -> tuple[int, int]:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return 0, 0
    return int(columns.max() - columns.min() + 1), int(rows.max() - rows.min() + 1)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    result = np.quantile(array, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "min": float(result[0]),
        "p10": float(result[1]),
        "p25": float(result[2]),
        "median": float(result[3]),
        "p75": float(result[4]),
        "p90": float(result[5]),
        "max": float(result[6]),
    }


def load_c2_rows(csv_path: Path) -> list[tuple[str, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageId", "ClassId", "EncodedPixels"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"train.csv needs columns {sorted(required)}")
        for row in reader:
            encoded = (row.get("EncodedPixels") or "").strip()
            if str(row.get("ClassId")) == "2" and encoded:
                rows.append((str(row["ImageId"]), encoded))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    c2_rows = load_c2_rows(args.data_dir / "train.csv")
    records = []
    for image_id, encoded in c2_rows:
        original = rle_decode(encoded)
        resized = resize_mask_like_opencv_nearest(original)
        original_areas = component_areas(original)
        resized_areas = component_areas(resized)
        original_pixels = int(original.sum())
        resized_pixels = int(resized.sum())
        expected_pixels = original_pixels * (MODEL_WIDTH / ORIGINAL_WIDTH)
        normalized_retention = resized_pixels / expected_pixels if expected_pixels else 0.0
        original_bbox_width, original_bbox_height = bounding_box(original)
        resized_bbox_width, resized_bbox_height = bounding_box(resized)
        records.append(
            {
                "image_id": image_id,
                "original_pixels": original_pixels,
                "resized_pixels": resized_pixels,
                "raw_pixel_ratio": resized_pixels / original_pixels,
                "normalized_pixel_retention": normalized_retention,
                "vanished_after_resize": int(resized_pixels == 0),
                "original_components": int(original_areas.size),
                "resized_components": int(resized_areas.size),
                "components_lost": max(0, int(original_areas.size - resized_areas.size)),
                "original_smallest_component": int(original_areas.min()),
                "resized_smallest_component": int(resized_areas.min()) if resized_areas.size else 0,
                "original_bbox_width": original_bbox_width,
                "original_bbox_height": original_bbox_height,
                "resized_bbox_width": resized_bbox_width,
                "resized_bbox_height": resized_bbox_height,
            }
        )

    if not records:
        raise RuntimeError("No positive C2 annotations found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = args.output_dir / "c2_resolution_audit.csv"
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    retention = [record["normalized_pixel_retention"] for record in records]
    original_pixels = [record["original_pixels"] for record in records]
    resized_pixels = [record["resized_pixels"] for record in records]
    sorted_worst = sorted(records, key=lambda record: record["normalized_pixel_retention"])
    summary = {
        "positive_c2_images": len(records),
        "resize": "256x1600 -> 256x800, nearest-neighbor",
        "vanished_masks": sum(record["vanished_after_resize"] for record in records),
        "images_losing_components": sum(record["components_lost"] > 0 for record in records),
        "total_components_before": sum(record["original_components"] for record in records),
        "total_components_after": sum(record["resized_components"] for record in records),
        "normalized_retention_below_0_5": sum(value < 0.5 for value in retention),
        "normalized_retention_below_0_75": sum(value < 0.75 for value in retention),
        "original_pixels": quantiles(original_pixels),
        "resized_pixels": quantiles(resized_pixels),
        "normalized_pixel_retention": quantiles(retention),
        "worst_10_images": [
            {
                "image_id": record["image_id"],
                "original_pixels": record["original_pixels"],
                "resized_pixels": record["resized_pixels"],
                "normalized_pixel_retention": record["normalized_pixel_retention"],
                "components_lost": record["components_lost"],
            }
            for record in sorted_worst[:10]
        ],
        "interpretation_note": (
            "A raw pixel ratio near 0.5 is expected because width is halved. "
            "Normalized retention divides by this expected 0.5 scale; values near 1 mean area was preserved as expected."
        ),
    }
    json_output = args.output_dir / "c2_resolution_audit_summary.json"
    json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {csv_output}")
    print(f"wrote {json_output}")


if __name__ == "__main__":
    main()
