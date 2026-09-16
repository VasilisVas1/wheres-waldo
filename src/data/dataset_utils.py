"""Build a manifest of the raw dataset: one row per image, with its Roboflow
split, its underlying scene id, and its Waldo bounding box(es).

The Roboflow export names files like `10_15_4_jpg.rf.<hash>.jpg` — `10_15_4`
is the id of the original puzzle-book scene, and `.rf.<hash>` marks one of
several augmented copies Roboflow generated from it (flip / rotate / noise).
Grouping by that scene id is what makes it possible to verify the provided
train/valid/test split doesn't leak the same underlying scene across splits.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data.voc_parser import parse_voc_xml

SCENE_ID_PATTERN = re.compile(r"^(.*)_jpg\.rf\.[0-9a-f]+$")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "wheres-waldo-voc"


def scene_id_from_stem(file_stem: str) -> str:
    """Recover the original scene id from a Roboflow-exported file stem.

    e.g. "10_15_4_jpg.rf.2df611cadb1226890b230a5b4d6d08e3" -> "10_15_4"
    Falls back to the full stem if the expected Roboflow suffix pattern
    isn't present (e.g. for images from a different source later).
    """
    match = SCENE_ID_PATTERN.match(file_stem)
    return match.group(1) if match else file_stem


def build_manifest(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    """Scan train/valid/test folders and return one row per image:
    split, scene_id, image_path, xml_path, image width/height, and the
    list of Waldo boxes (in absolute pixel coordinates) found in it.
    """
    dataset_dir = Path(dataset_dir)
    rows = []

    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        for xml_path in sorted(split_dir.glob("*.xml")):
            ann = parse_voc_xml(xml_path)
            image_path = split_dir / ann.filename
            if not image_path.exists():
                # Roboflow sometimes stores the extension case differently
                candidates = list(split_dir.glob(xml_path.stem + ".*"))
                image_path = next(p for p in candidates if p.suffix != ".xml")

            rows.append(
                {
                    "split": split,
                    "scene_id": scene_id_from_stem(xml_path.stem),
                    "file_stem": xml_path.stem,
                    "image_path": str(image_path),
                    "xml_path": str(xml_path),
                    "width": ann.width,
                    "height": ann.height,
                    "num_boxes": len(ann.boxes),
                    "boxes": [b.as_tuple() for b in ann.boxes],
                }
            )

    return pd.DataFrame(rows)
