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
