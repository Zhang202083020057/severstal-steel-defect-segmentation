"""Train U-Net or DeepLabV3+ on the Severstal dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from steel_common import (
    NUM_CLASSES,
    SegmentationLoss,
    SteelDataset,
    build_c2_transforms,
    build_model,
    build_transforms,
    dice_sums,
    load_annotation_table,
    make_targets,
    seed_everything,
    split_ids,
)


def subset_ids(ids: list[str], maximum: int, seed: int) -> list[str]:
    if maximum <= 0 or maximum >= len(ids):
        return ids
    generator = np.random.default_rng(seed)
    positions = generator.choice(len(ids), size=maximum, replace=False)
    return [ids[position] for position in positions]


def class_counts(table: pd.DataFrame, ids: list[str]) -> list[int]:
    return make_targets(table.loc[ids]).sum(axis=0).astype(int).tolist()


def run_epoch(model, loader, criterion, device, optimizer, scaler, thresholds):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    item_count = 0
    dice_sum = torch.zeros(NUM_CLASSES, device=device)
    dice_count = torch.zeros(NUM_CLASSES, device=device)
    progress = tqdm(loader, desc="train" if training else "valid", leave=False)

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        item_count += batch_size
        batch_sum, batch_count = dice_sums(logits.detach(), masks, thresholds)
        dice_sum += batch_sum
        dice_count += batch_count
        progress.set_postfix(loss=f"{loss.item():.4f}")

    class_dice = (dice_sum / dice_count.clamp_min(1)).cpu().numpy()
    return loss_sum / max(item_count, 1), class_dice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=["unet", "fpn", "deeplabv3plus"], default="unet"
    )
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--loss", choices=["bce_dice", "focal_dice"], default="bce_dice")
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="basic")
    parser.add_argument(
        "--c2-augmentation",
        action="store_true",
        help=(
            "Apply stronger online transforms only to training images containing Class 2; "
            "this does not oversample or create new image files"
        ),
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--thresholds", default="0.5,0.5,0.5,0.5")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This AutoDL training script expects a CUDA GPU. Run check_environment.py first.")
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = 2
        args.height = 64
        args.width = 400
        args.max_train_samples = 32
        args.max_val_samples = 16
        args.encoder_weights = "none"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if len(thresholds) != NUM_CLASSES:
        raise ValueError("--thresholds needs four comma-separated values")

    table = load_annotation_table(args.data_dir)
    train_ids, val_ids = split_ids(table, args.val_size, args.seed)
    train_ids = subset_ids(train_ids, args.max_train_samples, args.seed)
    val_ids = subset_ids(val_ids, args.max_val_samples, args.seed + 1)

    train_transform = build_transforms(
        args.height, args.width, training=True, strong=args.augmentation == "strong"
    )
    c2_transform = (
        build_c2_transforms(args.height, args.width) if args.c2_augmentation else None
    )
    valid_transform = build_transforms(args.height, args.width, training=False, strong=False)
    train_dataset = SteelDataset(
        args.data_dir / "train_images",
        train_ids,
        train_transform,
        table,
        c2_transform=c2_transform,
    )
    valid_dataset = SteelDataset(
        args.data_dir / "train_images", val_ids, valid_transform, table
    )
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)

    weights = None if args.encoder_weights == "none" else args.encoder_weights
    model = build_model(args.architecture, args.encoder, weights).cuda()
    criterion = SegmentationLoss(args.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # torch.cuda.amp keeps compatibility with the PyTorch versions commonly
    # preinstalled in AutoDL images; it is equivalent to CUDA GradScaler.
    scaler = torch.cuda.amp.GradScaler()

    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"train={len(train_ids)} val={len(val_ids)}")
    print(f"positive masks train={class_counts(table, train_ids)} val={class_counts(table, val_ids)}")
    print(
        f"model={args.architecture}/{args.encoder}, loss={args.loss}, "
        f"aug={args.augmentation}, c2_augmentation={args.c2_augmentation}"
    )

    best_score = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(
            model, train_loader, criterion, torch.device("cuda"), optimizer, scaler, thresholds
        )
        with torch.no_grad():
            val_loss, val_dice = run_epoch(
                model, valid_loader, criterion, torch.device("cuda"), None, scaler, thresholds
            )
        scheduler.step()
        score = float(val_dice.mean())
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": float(train_dice.mean()),
            "val_dice": score,
            **{f"val_dice_class_{index + 1}": float(value) for index, value in enumerate(val_dice)},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": args.architecture,
                    "encoder": args.encoder,
                    "height": args.height,
                    "width": args.width,
                    "thresholds": thresholds,
                    "best_val_dice": best_score,
                },
                args.output_dir / "best_model.pt",
            )
            print(f"saved new best model, Dice={best_score:.6f}")


if __name__ == "__main__":
    main()
