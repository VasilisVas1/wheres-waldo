"""Load Hey-Waldo's pre-cropped, human-curated classification patches as
extra training data for the stage-1 sliding-window classifier.

Hey-Waldo ships 64/128/256px patches (each a tile cut from one of the same
19 original scenes we already use). Filenames encode provenance directly:
`10_15_4.jpg` is scene 10, grid tile (15, 4) — the leading number is the
scene id, which lets us respect the exact same fold assignment
(`data/processed/fold_assignment.csv`) used everywhere else, so folding this
data in never leaks a held-out validation scene into training.

All three scales are resized down to PATCH_SIZE (64) rather than using only
the native 64px folder: this isn't just "more data", it mimics exactly what
our sliding-window inference does (crop a window, resize it to 64x64, then
classify) at each of its window sizes, so a resized-down 128px or 256px
patch is realistic training signal for what a coarser sliding window will
actually feed the classifier at inference time.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.patch_generation import PATCH_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEY_WALDO_DIR = PROJECT_ROOT / "Hey-Waldo"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "hey_waldo"

SCENE_ID_PATTERN = re.compile(r"^(\d+)_")


def _scene_id_from_filename(filename: str) -> str:
    match = SCENE_ID_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unexpected Hey-Waldo filename format: {filename}")
    return match.group(1)


def build_hey_waldo_manifest(
    scales: tuple[str, ...] = ("64", "128", "256"),
    max_negatives: int = 3000,
    seed: int = 0,
) -> pd.DataFrame:
    """Resize Hey-Waldo patches to PATCH_SIZE and materialize them under
    data/processed/patches/hey_waldo/, returning a manifest with the same
    columns as patch_manifest.csv (scene_id, label, path) so it can simply
    be concatenated with it.
    """
    rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    positive_records = []
    negative_records = []

    for scale in scales:
        for label_name, label in (("waldo", 1), ("notwaldo", 0)):
            src_dir = HEY_WALDO_DIR / scale / label_name
            if not src_dir.exists():
                continue
            for src_path in src_dir.iterdir():
                if not src_path.is_file():
                    continue
                scene_id = _scene_id_from_filename(src_path.name)
                record = {"scene_id": scene_id, "label": label, "_scale": scale, "_src": src_path}
                (positive_records if label == 1 else negative_records).append(record)

    # Hey-Waldo's patch folders also include a couple of scenes (ids 20, 21)
    # that aren't among our 19 verified clean scenes and have no fold
    # assignment — drop them rather than risk an unaccounted-for leak.
    known_scene_ids = {str(i) for i in range(1, 20)}
    positive_records = [r for r in positive_records if r["scene_id"] in known_scene_ids]
    negative_records = [r for r in negative_records if r["scene_id"] in known_scene_ids]

    if len(negative_records) > max_negatives:
        keep_idx = rng.choice(len(negative_records), size=max_negatives, replace=False)
        negative_records = [negative_records[i] for i in keep_idx]

    rows = []
    for record in positive_records + negative_records:
        label_name = "waldo" if record["label"] == 1 else "notwaldo"
        out_name = f"{record['scene_id']}_{record['_scale']}_{label_name}_{record['_src'].name}"
        out_path = OUT_DIR / out_name

        img = Image.open(record["_src"]).convert("RGB").resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
        img.save(out_path)

        rows.append({"scene_id": record["scene_id"], "label": record["label"], "path": str(out_path)})

    return pd.DataFrame(rows)
