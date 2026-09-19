"""Native-resolution tiling for YOLO.

Why: Roboflow's export stretched every scene to 640x640, so Waldo shrank to ~15x25 px in a 2048-px-wide
scan (about 50x75 px natively) and YOLO had almost nothing to look at. Here we keep the native pixels and
show YOLO 640x640 *tiles* instead of a whole downscaled scene:

* training: tiles are sampled around each Waldo (at random offsets), plus background-only tiles and tiles
  with a real Waldo crop from another training scene pasted in;
* inference: the scene is covered by an overlapping grid of tiles, YOLO runs on each, and the tile
  detections are mapped back to scene coordinates and merged with our own NMS.

Everything here is fold-aware by construction: callers pass only a fold's *training* scenes to
`export_tiled_train`, so no held-out scene (or crop of one) can leak into training.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.synthetic_compositing import composite_synthetic_scene, extract_crop
from src.eval.box_utils import non_max_suppression

TILE = 640
OVERLAP = 240  # >= the largest Waldo box (~215 px), so every Waldo lies fully inside at least one tile


def tile_grid(width: int, height: int, tile: int = TILE, overlap: int = OVERLAP) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the whole image; the last row/column is shifted back to stay in bounds."""
    stride = tile - overlap

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile, stride))
        s.append(size - tile)
        return s

    return [(x, y, x + tile, y + tile) for y in starts(height) for x in starts(width)]


def build_native_crop_pool(native_manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Waldo crops at full native resolution (soft alpha edge), one per annotated box."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = []
    for _, scene in native_manifest.iterrows():
        image = Image.open(scene["image_path"]).convert("RGB")
        for i, box in enumerate(scene["boxes"]):
            path = out_dir / f"{scene['scene_id']}_{i}.png"
            extract_crop(image, tuple(box)).save(path)
            rows.append({"scene_id": str(scene["scene_id"]), "path": str(path)})
    return pd.DataFrame(rows)


def _visible_fraction(box, tile_box) -> float:
    bx1, by1, bx2, by2 = box
    tx1, ty1, tx2, ty2 = tile_box
    iw = max(0, min(bx2, tx2) - max(bx1, tx1))
    ih = max(0, min(by2, ty2) - max(by1, ty1))
    area = (bx2 - bx1) * (by2 - by1)
    return (iw * ih) / area if area > 0 else 0.0


def _tile_labels(boxes, tile_box):
    """Boxes fully inside the tile in tile coordinates, or None if any box is only partly visible
    (a half-cut Waldo would be an unlabeled near-positive, so such tiles are rejected)."""
    tx1, ty1, _, _ = tile_box
    out = []
    for box in boxes:
        frac = _visible_fraction(box, tile_box)
        if frac == 0:
            continue
        if frac < 0.999:
            return None
        out.append((box[0] - tx1, box[1] - ty1, box[2] - tx1, box[3] - ty1))
    return out


def _write_label(path: Path, boxes, size: int = TILE) -> None:
    lines = [
        f"0 {((x1 + x2) / 2) / size:.6f} {((y1 + y2) / 2) / size:.6f} {(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}"
        for (x1, y1, x2, y2) in boxes
    ]
    path.write_text("\n".join(lines))


def _save(tile_img: Image.Image, boxes, images_dir: Path, labels_dir: Path, name: str) -> None:
    tile_img.convert("RGB").save(images_dir / f"{name}.jpg", quality=95)
    _write_label(labels_dir / f"{name}.txt", boxes)


def export_tiled_train(
    train_scenes: pd.DataFrame,
    native_crop_pool: pd.DataFrame,
    out_dir: Path,
    n_pos_per_box: int = 10,
    n_neg_per_scene: int = 6,
    n_synth_per_scene: int = 6,
    seed: int = 0,
) -> dict[str, int]:
    """Write a tiled YOLO training set (from `train_scenes` only) and a data.yaml; returns tile counts.

    `train_scenes` needs scene_id, image_path, width, height, boxes (native pixel coordinates).
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir, labels_dir = out_dir / "train" / "images", out_dir / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    pool = native_crop_pool[native_crop_pool["scene_id"].isin(set(train_scenes["scene_id"]))]
    counts = {"positive": 0, "negative": 0, "synthetic": 0}

    def random_tile(W, H):
        x, y = int(rng.integers(0, W - TILE + 1)), int(rng.integers(0, H - TILE + 1))
        return (x, y, x + TILE, y + TILE)

    for _, scene in train_scenes.iterrows():
        sid, W, H = str(scene["scene_id"]), int(scene["width"]), int(scene["height"])
        boxes = [tuple(b) for b in scene["boxes"]]
        image = Image.open(scene["image_path"]).convert("RGB")
        other_crops = [Image.open(p) for p in pool[pool["scene_id"] != sid]["path"]]

        # 1) real positives: tiles containing each Waldo at a random offset
        for b_idx, (x1, y1, x2, y2) in enumerate(boxes):
            for i in range(n_pos_per_box):
                for _try in range(30):
                    lo_x, hi_x = max(0, x2 - TILE), min(x1, W - TILE)
                    lo_y, hi_y = max(0, y2 - TILE), min(y1, H - TILE)
                    ox, oy = int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1))
                    tile_box = (ox, oy, ox + TILE, oy + TILE)
                    labels = _tile_labels(boxes, tile_box)
                    if labels is not None:
                        _save(image.crop(tile_box), labels, images_dir, labels_dir, f"{sid}_pos{b_idx}_{i:02d}")
                        counts["positive"] += 1
                        break

        # 2) background-only tiles (these carry the hard negatives: other stripes, faces, glasses...)
        made = 0
        for _try in range(200):
            if made >= n_neg_per_scene:
                break
            tile_box = random_tile(W, H)
            if _tile_labels(boxes, tile_box) == []:
                _save(image.crop(tile_box), [], images_dir, labels_dir, f"{sid}_neg{made:02d}")
                counts["negative"] += 1
                made += 1

        # 3) synthetic: a real Waldo crop from another training scene pasted onto a background-only tile
        made = 0
        for _try in range(200):
            if made >= n_synth_per_scene or not other_crops:
                break
            tile_box = random_tile(W, H)
            if _tile_labels(boxes, tile_box) != []:
                continue
            composite, new_boxes = composite_synthetic_scene(
                image.crop(tile_box), [], other_crops, rng, n_paste=int(rng.integers(1, 3))
            )
            if new_boxes:
                _save(composite, new_boxes, images_dir, labels_dir, f"{sid}_synth{made:02d}")
                counts["synthetic"] += 1
                made += 1

    # ultralytics needs a val split; give it a few *training* tiles so no held-out scene is touched in training
    # (checkpoint = last epoch, no selection on held-out data)
    val_images, val_labels = out_dir / "val" / "images", out_dir / "val" / "labels"
    val_images.mkdir(parents=True)
    val_labels.mkdir(parents=True)
    for p in sorted(images_dir.glob("*_pos*.jpg"))[:24]:
        shutil.copy(p, val_images / p.name)
        shutil.copy(labels_dir / f"{p.stem}.txt", val_labels / f"{p.stem}.txt")

    (out_dir / "data.yaml").write_text(
        yaml.safe_dump({"path": str(out_dir.resolve()), "train": "train/images", "val": "val/images", "names": {0: "waldo"}})
    )
    return counts


def predict_tiled(
    model,
    image: Image.Image,
    conf: float = 0.05,
    tile: int = TILE,
    overlap: int = OVERLAP,
    batch: int = 8,
    nms_iou: float = 0.4,
    device="cpu",
) -> list[tuple[tuple[float, float, float, float], float]]:
    """Run `model` over an overlapping tile grid and merge duplicate detections across tiles with our own NMS.

    Returns [(box_xyxy_in_image_coords, confidence), ...] sorted by confidence, highest first.
    """
    image = image.convert("RGB")
    tiles = tile_grid(image.width, image.height, tile, overlap)
    boxes, scores = [], []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i : i + batch]
        results = model.predict([image.crop(t) for t in chunk], imgsz=tile, conf=conf, verbose=False, device=device)
        for (tx1, ty1, _, _), result in zip(chunk, results):
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1))
                scores.append(float(b.conf.item()))
    if not boxes:
        return []
    keep = non_max_suppression(np.array(boxes), np.array(scores), iou_threshold=nms_iou)
    kept = [(tuple(boxes[j]), scores[j]) for j in keep]
    return sorted(kept, key=lambda d: -d[1])
