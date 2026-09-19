"""Build a manifest using Hey-Waldo's native-resolution original scans instead
of the 640x640 images Roboflow exported.

Roboflow's export applied "Resize: Stretch to 640x640" (a non-uniform,
per-axis stretch) and, for some scenes, also flipped or rotated the page.
Hey-Waldo's `original-images/<scene_id>.jpg` files are the same 19 scenes
at their true native resolution and orientation.

So a Roboflow box cannot be mapped to a native box by rescaling alone: for 8
of the 19 scenes (7, 9, 11, 14, 15, 16, 17, 18) the Roboflow image is a
mirrored/rotated version of the native scan, and a plain rescale lands the
box on unrelated content (a policeman, a letter, a red panel...). The fix
is to find, per scene, which of the 8 flips/rotations of the native scan
reproduces the Roboflow image (image correlation is ~0.99 for the right one
and <=0.75 for every wrong one), then map the box back through the inverse
of that transform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEY_WALDO_ORIGINALS = PROJECT_ROOT / "Hey-Waldo" / "original-images"

_T = Image.Transpose
# PIL transform that turns the native scan into the Roboflow orientation
ORIENTATIONS: dict[str, "Image.Transpose | None"] = {
    "identity": None,
    "flip_lr": _T.FLIP_LEFT_RIGHT,
    "flip_tb": _T.FLIP_TOP_BOTTOM,
    "rot180": _T.ROTATE_180,
    "rot90": _T.ROTATE_90,
    "rot270": _T.ROTATE_270,
    "transpose": _T.TRANSPOSE,
    "transverse": _T.TRANSVERSE,
}
_SWAPS_AXES = {"rot90", "rot270", "transpose", "transverse"}
MIN_SCORE = 0.9  # correlation of the right orientation is ~0.99
MIN_MARGIN = 0.1  # ...and the runner-up is far lower


def _signature(image: Image.Image, size: int = 96) -> np.ndarray:
    a = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def find_orientation(roboflow_image: Image.Image, native_image: Image.Image) -> tuple[str, float, float]:
    """Return (orientation name, correlation of best, correlation of runner-up)."""
    target = _signature(roboflow_image)
    scores = {}
    for name, op in ORIENTATIONS.items():
        candidate = native_image if op is None else native_image.transpose(op)
        scores[name] = float(target @ _signature(candidate))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[0][1], ranked[1][1]


def roboflow_box_to_native(
    box: tuple[float, float, float, float],
    orientation: str,
    roboflow_size: tuple[int, int],
    native_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map an (x1, y1, x2, y2) box from the Roboflow image into native-scan pixels."""
    W, H = native_size
    tw, th = (H, W) if orientation in _SWAPS_AXES else (W, H)  # size of the native scan after the transform
    rw, rh = roboflow_size
    u1, v1, u2, v2 = box[0] * tw / rw, box[1] * th / rh, box[2] * tw / rw, box[3] * th / rh

    def to_native(u: float, v: float) -> tuple[float, float]:
        return {
            "identity": (u, v),
            "flip_lr": (W - u, v),
            "flip_tb": (u, H - v),
            "rot180": (W - u, H - v),
            "rot90": (W - v, u),
            "rot270": (v, H - u),
            "transpose": (v, u),
            "transverse": (W - v, H - u),
        }[orientation]

    (xa, ya), (xb, yb) = to_native(u1, v1), to_native(u2, v2)
    return (round(min(xa, xb)), round(min(ya, yb)), round(max(xa, xb)), round(max(ya, yb)))


def build_native_manifest(clean_manifest: pd.DataFrame) -> pd.DataFrame:
    """Given the Roboflow-space clean manifest, return an equivalent manifest pointing at the native scans
    with boxes mapped into native pixel coordinates (orientation-corrected, see module docstring)."""
    rows = []
    for _, row in clean_manifest.iterrows():
        scene_id = str(row["scene_id"])
        native_path = HEY_WALDO_ORIGINALS / f"{scene_id}.jpg"
        if not native_path.exists():
            raise FileNotFoundError(f"No native image for scene {scene_id} at {native_path}")

        native_img = Image.open(native_path).convert("RGB")
        roboflow_img = Image.open(row["image_path"])
        orientation, score, runner_up = find_orientation(roboflow_img, native_img)
        if score < MIN_SCORE or score - runner_up < MIN_MARGIN:
            raise ValueError(
                f"Scene {scene_id}: cannot confidently align Roboflow image to the native scan "
                f"(best {orientation} {score:.3f}, runner-up {runner_up:.3f})"
            )

        native_boxes = [
            roboflow_box_to_native(tuple(b), orientation, (row["width"], row["height"]), native_img.size)
            for b in row["boxes"]
        ]
        rows.append(
            {
                "scene_id": scene_id,
                "split": row["split"],
                "image_path": str(native_path),
                "width": native_img.width,
                "height": native_img.height,
                "num_boxes": len(native_boxes),
                "boxes": native_boxes,
                "orientation": orientation,
                "alignment_score": round(score, 3),
            }
        )

    return pd.DataFrame(rows)
