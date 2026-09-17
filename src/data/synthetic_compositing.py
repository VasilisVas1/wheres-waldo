"""Copy-paste synthetic augmentation: multiply the number of distinct
full-scene training images for YOLO by pasting real Waldo crops (taken from
*other* training scenes) onto a background scene at new random positions,
scales, and rotations.

Why this is the real lever for YOLO's problem (see notebook 04/05): YOLO
wasn't short on augmentation — mosaic/flip/HSV jitter were already on — it
was short on distinct *scenes*. 15 training images is too few no matter how
each one is perturbed. Copy-paste generates new (background, box) pairs
rather than new views of the same 15 pairs, which is a qualitatively
different kind of diversity.

This is fold-aware by construction: callers pass only that fold's *training*
scenes as both the crop source and the paste target, so no information about
held-out validation scenes ever leaks into a synthetic training image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from src.eval.box_utils import iou

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CROP_POOL_DIR = PROJECT_ROOT / "data" / "processed" / "crop_pool"


@dataclass
class WaldoCrop:
    image: Image.Image  # RGBA, tight crop around Waldo with a soft alpha edge
    source_scene_id: str


def extract_crop(scene_image: Image.Image, box: tuple[int, int, int, int], feather_px: int = 2) -> Image.Image:
    """Crop a box out of a scene with a small soft-edged alpha mask, so a
    paste doesn't leave an obvious hard rectangular seam.
    """
    x1, y1, x2, y2 = box
    crop = scene_image.crop((x1, y1, x2, y2)).convert("RGBA")

    mask = Image.new("L", crop.size, 255)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather_px))
        # re-darken the border so the feather actually fades to transparent
        border = Image.new("L", crop.size, 0)
        inner = (feather_px, feather_px, crop.size[0] - feather_px, crop.size[1] - feather_px)
        if inner[2] > inner[0] and inner[3] > inner[1]:
            border.paste(255, inner)
            border = border.filter(ImageFilter.GaussianBlur(feather_px))
            mask = Image.composite(mask, border, border)
    crop.putalpha(mask)
    return crop


def _random_transform(crop: Image.Image, rng: np.random.Generator) -> Image.Image:
    scale = rng.uniform(0.8, 1.25)
    new_size = (max(4, int(crop.width * scale)), max(4, int(crop.height * scale)))
    out = crop.resize(new_size, Image.BILINEAR)

    angle = rng.uniform(-12, 12)
    out = out.rotate(angle, expand=True, resample=Image.BILINEAR)

    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    brightness = rng.uniform(0.85, 1.15)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    return out


def composite_synthetic_scene(
    background: Image.Image,
    existing_boxes: list[tuple[int, int, int, int]],
    paste_crops: list[Image.Image],
    rng: np.random.Generator,
    n_paste: int = 1,
    min_iou_gap: float = 0.0,
) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """Paste `n_paste` random crops onto `background` at random positions
    that don't overlap each other or the scene's existing real box(es).

    Returns the composited image (RGB) and the full list of boxes (existing
    real boxes + newly pasted ones).
    """
    canvas = background.convert("RGBA")
    W, H = canvas.size
    new_boxes = list(existing_boxes)

    for _ in range(n_paste):
        crop = _random_transform(paste_crops[rng.integers(0, len(paste_crops))], rng)
        cw, ch = crop.size
        if cw >= W or ch >= H:
            continue

        for _attempt in range(20):
            x = int(rng.integers(0, W - cw))
            y = int(rng.integers(0, H - ch))
            candidate = (x, y, x + cw, y + ch)
            if all(iou(np.array(candidate), np.array(b)) <= min_iou_gap for b in new_boxes):
                canvas.alpha_composite(crop, dest=(x, y))
                new_boxes.append(candidate)
                break

    return canvas.convert("RGB"), new_boxes


def build_crop_pool(clean_manifest: pd.DataFrame, native_manifest: pd.DataFrame) -> pd.DataFrame:
    """Extract every scene's Waldo box as a soft-edged crop from the sharper
    native-resolution image, resized down to the box's size in our working
    640x640 space, and save it to disk.

    Returns a manifest (scene_id, path) so callers can filter crops to a
    given fold's training scenes before using them as paste sources.
    """
    CROP_POOL_DIR.mkdir(parents=True, exist_ok=True)
    native_by_scene = {row["scene_id"]: row for _, row in native_manifest.iterrows()}

    rows = []
    for _, row in clean_manifest.iterrows():
        scene_id = row["scene_id"]
        native_row = native_by_scene[scene_id]
        native_img = Image.open(native_row["image_path"])

        for i, (box_640, native_box) in enumerate(zip(row["boxes"], native_row["boxes"])):
            target_w = box_640[2] - box_640[0]
            target_h = box_640[3] - box_640[1]

            crop = extract_crop(native_img, native_box)
            crop = crop.resize((max(4, target_w), max(4, target_h)), Image.LANCZOS)

            out_path = CROP_POOL_DIR / f"{scene_id}_{i}.png"
            crop.save(out_path)
            rows.append({"scene_id": scene_id, "path": str(out_path)})

    return pd.DataFrame(rows)
