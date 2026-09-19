"""Orientation detection + box mapping between Roboflow images and native scans (src/data/native_manifest.py)."""

import numpy as np
import pytest
from PIL import Image

from src.data.native_manifest import ORIENTATIONS, find_orientation, roboflow_box_to_native

W, H = 300, 200
NATIVE_BOX = (40, 60, 90, 130)


def _native_scan() -> Image.Image:
    rng = np.random.default_rng(0)
    pixels = (rng.random((H, W, 3)) * 180).astype(np.uint8)  # noise stays below the marker's red
    pixels[NATIVE_BOX[1] : NATIVE_BOX[3], NATIVE_BOX[0] : NATIVE_BOX[2]] = [255, 0, 0]
    return Image.fromarray(pixels)


@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
def test_orientation_is_detected_and_box_maps_back(orientation):
    native = _native_scan()
    op = ORIENTATIONS[orientation]
    roboflow = (native if op is None else native.transpose(op)).resize((640, 640), Image.BILINEAR)

    ys, xs = np.where(np.asarray(roboflow)[..., 0] > 215)  # where the marker ended up in the Roboflow image
    roboflow_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    detected, score, runner_up = find_orientation(roboflow, native)
    assert detected == orientation
    assert score - runner_up > 0.1

    mapped = roboflow_box_to_native(roboflow_box, orientation, (640, 640), native.size)
    assert max(abs(np.array(mapped) - np.array(NATIVE_BOX))) <= 3
