"""Convert the clean scene manifest + a fold assignment into the directory
layout `ultralytics` expects: images/ and labels/ folders with one .txt per
image (YOLO format: `class cx cy w h`, all normalized 0-1), plus a data.yaml.

Kept separate from the sliding-window patch pipeline (src/data/patch_generation.py)
since YOLO trains on full scenes with normalized box coordinates, not
pre-cropped classification patches.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def _write_yolo_label(path: Path, boxes: list[tuple[int, int, int, int]], img_w: int, img_h: int) -> None:
    lines = []
    for (x1, y1, x2, y2) in boxes:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))


def export_yolo_fold(
    scenes: pd.DataFrame, fold_assignment: pd.DataFrame, val_fold: int, out_dir: Path
) -> Path:
    """Write train/val images+labels for one fold to out_dir, return the data.yaml path."""
    if out_dir.exists():
        shutil.rmtree(out_dir)

    merged = scenes.merge(fold_assignment, on="scene_id", how="left")

    for split, subset in (("train", merged[merged["val_fold"] != val_fold]), ("val", merged[merged["val_fold"] == val_fold])):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        for _, row in subset.iterrows():
            image = Image.open(row["image_path"]).convert("RGB")
            dest_img = out_dir / split / "images" / f"{row['scene_id']}.jpg"
            image.save(dest_img)
            _write_yolo_label(
                out_dir / split / "labels" / f"{row['scene_id']}.txt", row["boxes"], row["width"], row["height"]
            )

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "waldo"},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml))
    return yaml_path


def add_synthetic_training_images(
    train_scenes: pd.DataFrame,
    crop_pool: pd.DataFrame,
    out_dir: Path,
    n_per_scene: int = 10,
    n_paste: int = 1,
    seed: int = 0,
) -> int:
    """Generate copy-paste synthetic composites from a fold's *training*
    scenes only, and add them into out_dir/train/{images,labels} alongside
    the real ones `export_yolo_fold` already wrote there.

    Both the backgrounds and the pasted crops come exclusively from
    `train_scenes` / the matching rows of `crop_pool` — the caller is
    responsible for pre-filtering both to one fold's training split, so
    nothing about a held-out validation scene can leak in here.

    Returns the number of synthetic images written.
    """
    from src.data.synthetic_compositing import composite_synthetic_scene

    rng = np.random.default_rng(seed)
    train_scene_ids = set(train_scenes["scene_id"])
    pool = crop_pool[crop_pool["scene_id"].isin(train_scene_ids)]

    images_dir = out_dir / "train" / "images"
    labels_dir = out_dir / "train" / "labels"
    n_written = 0

    for _, scene in train_scenes.iterrows():
        background = Image.open(scene["image_path"]).convert("RGB")
        other_crop_paths = pool[pool["scene_id"] != scene["scene_id"]]["path"].tolist()
        if not other_crop_paths:
            continue
        paste_crops = [Image.open(p) for p in other_crop_paths]

        for i in range(n_per_scene):
            composite, boxes = composite_synthetic_scene(
                background, scene["boxes"], paste_crops, rng, n_paste=n_paste
            )
            name = f"{scene['scene_id']}_synth{i:02d}"
            composite.save(images_dir / f"{name}.jpg")
            _write_yolo_label(labels_dir / f"{name}.txt", boxes, composite.width, composite.height)
            n_written += 1

    return n_written
