"""Build a manifest using Hey-Waldo's native-resolution original scans instead
of the 640x640 stretched images Roboflow exported.

Roboflow's export applied "Resize: Stretch to 640x640" (a non-uniform,
per-axis stretch — confirmed on the dataset version page) before we ever saw
the data. That's fine for a quick baseline, but it throws away resolution and
distorts aspect ratio right before the model has to find a ~20px object.
Hey-Waldo's `original-images/<scene_id>.jpg` files are the same 19 scenes at
their true native resolution (verified by inverse-transforming a known box
and confirming Waldo is exactly there).

Since the stretch was per-axis, the inverse transform is just independent
x/y scale factors: native_coord = roboflow_coord * (native_size / 640).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEY_WALDO_ORIGINALS = PROJECT_ROOT / "Hey-Waldo" / "original-images"


def build_native_manifest(clean_manifest: pd.DataFrame) -> pd.DataFrame:
    """Given the existing 640x640-space clean manifest, return an equivalent
    manifest pointing at Hey-Waldo's native-resolution images with boxes
    rescaled into native pixel coordinates.
    """
    rows = []
    for _, row in clean_manifest.iterrows():
        scene_id = str(row["scene_id"])
        native_path = HEY_WALDO_ORIGINALS / f"{scene_id}.jpg"
        if not native_path.exists():
            raise FileNotFoundError(f"No native image for scene {scene_id} at {native_path}")

        native_img = Image.open(native_path)
        nw, nh = native_img.size
        sx, sy = nw / row["width"], nh / row["height"]

        native_boxes = [
            (
                round(x1 * sx),
                round(y1 * sy),
                round(x2 * sx),
                round(y2 * sy),
            )
            for (x1, y1, x2, y2) in row["boxes"]
        ]

        rows.append(
            {
                "scene_id": scene_id,
                "split": row["split"],
                "image_path": str(native_path),
                "width": nw,
                "height": nh,
                "num_boxes": len(native_boxes),
                "boxes": native_boxes,
            }
        )

    return pd.DataFrame(rows)
