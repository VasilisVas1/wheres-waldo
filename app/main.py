"""Streamlit demo: upload a Where's Waldo page and get ranked candidates for where he is.

Three detectors to choose from: YOLO scanning native-resolution tiles at several scales (the best one, and
the default), the from-scratch sliding-window CNN (watch its heatmap build up live, then NMS collapse the
candidates), or YOLO on the whole shrunken scene. Each candidate is also shown as an enlarged close-up.

Run from the project root so `.streamlit/config.toml` (the theme) applies:  streamlit run app/main.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from waldo_theme import (
    apply_theme_html,
    hero_html,
    hidden_waldo_html,
    stripes_divider_html,
    verdict_html,
)

from src.data.patch_generation import PATCH_SIZE
from src.data.tiling import DEFAULT_SCALES, TILE, predict_multiscale, tile_grid
from src.eval.box_utils import non_max_suppression
from src.models.patch_classifier import PatchClassifier
from src.models.sliding_window import DEFAULT_WINDOW_SIZES, windows_for_scale
from src.data.patch_dataset import to_tensor

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Where's Waldo?", page_icon="🔍", layout="wide")
st.markdown(apply_theme_html(), unsafe_allow_html=True)
st.markdown(hero_html(), unsafe_allow_html=True)
st.caption(
    "Three ways to find Waldo: YOLO run over native-resolution tiles (best), a from-scratch sliding-window CNN, "
    "and YOLO on the whole shrunken scene. Scores from cross-validation on held-out scenes are in "
    "`notebooks/05` and `06`; Waldo is genuinely hard to find and none of these are perfect."
)


@st.cache_resource
def load_sliding_window_model() -> PatchClassifier | None:
    path = MODELS_DIR / "sliding_window_classifier.pt"
    if not path.exists():
        return None
    model = PatchClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_yolo_model():
    path = MODELS_DIR / "yolo_detector.pt"
    if not path.exists():
        return None
    from ultralytics import YOLO

    return YOLO(str(path))


@st.cache_resource
def load_tiled_yolo():
    path = MODELS_DIR / "yolo_tiled_detector.pt"
    if not path.exists():
        return None, {}
    from ultralytics import YOLO

    config_path = MODELS_DIR / "yolo_tiled_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return YOLO(str(path)), config


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_boxes(
    image: Image.Image,
    boxes_with_scores: list[tuple[tuple, float]],
    color: str,
    label_top: int | None = None,
) -> Image.Image:
    """Draw boxes sized for the image (a 2,000px scan needs far thicker lines than a thumbnail).

    Expects detections sorted best-first; the first `label_top` (default: all) get a "#rank score" tag, the
    rest are drawn as thin unlabelled boxes so a flood of low-value detections doesn't bury the good ones.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    line = max(4, image.width // 250)
    font = _font(max(18, image.width // 55))
    for rank, (box, score) in enumerate(boxes_with_scores, start=1):
        if label_top is not None and rank > label_top:
            draw.rectangle(box, outline=color, width=max(2, line // 2))
            continue
        draw.rectangle(box, outline=color, width=line)
        tag = f"#{rank}  {score * 100:.0f}%" if score is not None else f"#{rank}"
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        pad = 4
        tag_x = min(max(0, box[0]), max(0, image.width - (right - left) - 2 * pad))
        tag_y = box[1] - (bottom - top) - 2 * pad - line
        if tag_y < 0:  # no room above the box: put the tag below it instead
            tag_y = box[3] + line
        draw.rectangle((tag_x, tag_y, tag_x + right - left + 2 * pad, tag_y + bottom - top + 2 * pad), fill=color)
        draw.text((tag_x + pad - left, tag_y + pad - top), tag, fill="white", font=font)
    return out


def zoomed_crop(image: Image.Image, box: tuple, out_px: int = 360) -> Image.Image:
    """A square close-up around `box` with surrounding context, upscaled, with the exact box outlined."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1, 40) * 3, image.width, image.height)
    left = min(max(0, cx - side / 2), image.width - side)
    top = min(max(0, cy - side / 2), image.height - side)
    crop = image.crop((round(left), round(top), round(left + side), round(top + side))).resize(
        (out_px, out_px), Image.LANCZOS
    )
    scale = out_px / side
    ImageDraw.Draw(crop).rectangle(
        ((x1 - left) * scale, (y1 - top) * scale, (x2 - left) * scale, (y2 - top) * scale), outline="red", width=2
    )
    return crop


def show_candidates(image: Image.Image, detections: list[tuple[tuple, float]], max_show: int = 10) -> None:
    """Gallery of enlarged crops, one per detection, numbered like the boxes on the full scene."""
    if not detections:
        return
    shown = detections[:max_show]
    st.markdown(stripes_divider_html(), unsafe_allow_html=True)
    st.subheader("The usual suspects, up close")
    st.caption(
        "Each picture is an enlarged close-up of one box (red outline) with some surrounding context, numbered like "
        "the boxes above. Look for the red-and-white striped shirt and bobble hat, round glasses and a cane."
    )
    # the best guess gets double width; the others follow at normal width (empty columns just stay empty)
    layouts = [[2, 1, 1]] + [[1, 1, 1, 1]] * ((len(shown) - 3 + 3) // 4)
    rank = 0
    for spec in layouts:
        for col in st.columns(spec):
            if rank >= len(shown):
                break
            box, score = shown[rank]
            with col:
                st.image(
                    zoomed_crop(image, box, out_px=640 if rank == 0 else 420),
                    caption=f"#{rank + 1} — {score * 100:.0f}% confident" + (" (best guess)" if rank == 0 else ""),
                    use_container_width=True,
                )
            rank += 1


def run_sliding_window_live(model: PatchClassifier, image: Image.Image, score_threshold: float, placeholder):
    """Same multi-scale sliding-window logic as src/models/sliding_window.py,
    but updates a Streamlit placeholder as it goes so the search is visible
    rather than an instant answer.
    """
    W, H = image.width, image.height
    heat = np.zeros((H, W), dtype=np.float32)
    heat_count = np.zeros((H, W), dtype=np.float32)

    all_boxes, all_scores = [], []

    for size in DEFAULT_WINDOW_SIZES:
        stride = max(1, int(size * 0.5))
        boxes = windows_for_scale(W, H, size, stride)

        batch_size = 128
        for start in range(0, len(boxes), batch_size):
            batch = boxes[start : start + batch_size]
            crops = [image.crop(b).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR) for b in batch]
            tensors = torch.stack([to_tensor(np.array(c.convert("RGB"))) for c in crops])
            with torch.no_grad():
                scores = torch.sigmoid(model(tensors)).numpy()

            for box, score in zip(batch, scores):
                x1, y1, x2, y2 = box
                heat[y1:y2, x1:x2] += score
                heat_count[y1:y2, x1:x2] += 1
                if score >= score_threshold:
                    all_boxes.append(box)
                    all_scores.append(float(score))

            avg_heat = np.divide(heat, heat_count, out=np.zeros_like(heat), where=heat_count > 0)
            overlay = Image.fromarray((avg_heat * 255).astype(np.uint8)).convert("L")
            heat_rgb = Image.merge("RGB", (overlay, Image.new("L", overlay.size, 0), Image.new("L", overlay.size, 0)))
            blended = Image.blend(image.convert("RGB"), heat_rgb, alpha=0.5)
            placeholder.image(blended, caption=f"Searching... window size {size}px", use_container_width=True)
            time.sleep(0.03)

    if not all_boxes:
        return []
    keep_idx = non_max_suppression(np.array(all_boxes), np.array(all_scores), iou_threshold=0.2)
    return [(tuple(all_boxes[i]), all_scores[i]) for i in keep_idx]


sw_model = load_sliding_window_model()
yolo_model = load_yolo_model()
tiled_model, tiled_config = load_tiled_yolo()

if sw_model is None and yolo_model is None and tiled_model is None:
    st.warning(
        "No trained models found in `models/`. Run notebooks 03, 04 and 05 first to produce "
        "`sliding_window_classifier.pt`, `yolo_detector.pt` and `yolo_tiled_detector.pt`."
    )

uploaded = st.file_uploader("Drop in a page from a Where's Waldo book", type=["jpg", "jpeg", "png"])

TILED_NAME = "YOLO on native-resolution tiles"
available_models = [
    name
    for name, m in [
        (TILED_NAME, tiled_model),
        ("Sliding-window CNN (from scratch)", sw_model),
        ("YOLO on the whole shrunken scene", yolo_model),
    ]
    if m is not None
]
model_choice = st.radio("Who should do the looking?", available_models, horizontal=True) if available_models else None

if uploaded is not None and model_choice is not None:
    image = Image.open(uploaded).convert("RGB")
    shown_detections: list[tuple[tuple, float]] = []  # whatever the chosen model found, best first
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The page")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("The search")
        placeholder = st.empty()

        if model_choice == TILED_NAME:
            st.caption(
                f"Scans the image in overlapping {TILE}px tiles at three scales (0.75x, 1x, 1.5x), so Waldo keeps his pixels. "
                "Works best on full-size scans (1,300px+ wide); a small web image gives him too few pixels."
            )
            default_conf = float(tiled_config.get("conf", 0.05))
            conf = st.slider("Minimum confidence", 0.01, 0.95, min(max(default_conf, 0.01), 0.95), 0.01)
            top_n = st.slider("Show at most this many candidates", 1, 10, 3)
            scan_key = f"{uploaded.name}:{uploaded.size}"
            if st.button("Find Waldo!", type="primary"):
                scales = tuple(tiled_config.get("scales", DEFAULT_SCALES))
                n_tiles = sum(
                    len(tile_grid(round(image.width * s), round(image.height * s))) for s in scales
                )
                with st.spinner(f"Peering into the crowd... {n_tiles} tiles at {len(scales)} scales"):
                    # scan once at a low floor and keep it, so the sliders below re-filter instantly
                    st.session_state["tiled_scan"] = {
                        "key": scan_key,
                        "detections": predict_multiscale(tiled_model, image, scales=scales, conf=0.01),
                    }
            scan = st.session_state.get("tiled_scan")
            if scan and scan["key"] == scan_key:
                detections = [d for d in scan["detections"] if d[1] >= conf][:top_n]
                if detections:
                    st.markdown(verdict_html(detections[0][1]), unsafe_allow_html=True)
                    placeholder.image(
                        draw_boxes(image, detections, color="red"),
                        caption=f"{len(detections)} candidate(s), most confident first",
                        use_container_width=True,
                    )
                    shown_detections = detections
                else:
                    st.markdown(verdict_html(None), unsafe_allow_html=True)
        elif model_choice.startswith("Sliding-window"):
            score_threshold = st.slider("Detection confidence threshold", 0.1, 0.95, 0.6, 0.05)
            if st.button("Find Waldo!", type="primary"):
                detections = sorted(
                    run_sliding_window_live(sw_model, image, score_threshold, placeholder), key=lambda d: -d[1]
                )
                placeholder.image(
                    draw_boxes(image, detections, color="red", label_top=5),
                    caption=f"{len(detections)} detection(s) after NMS (the 5 most confident are numbered)",
                    use_container_width=True,
                )
                shown_detections = detections
        else:
            conf = st.slider("Detection confidence threshold", 0.05, 0.95, 0.1, 0.05)
            if st.button("Find Waldo!", type="primary"):
                yolo_result = yolo_model.predict(image, conf=conf, verbose=False)[0]
                detections = sorted(
                    ((tuple(b.xyxy[0].tolist()), b.conf.item()) for b in yolo_result.boxes), key=lambda d: -d[1]
                )
                placeholder.image(
                    draw_boxes(image, detections, color="red"),
                    caption=f"{len(detections)} detection(s)",
                    use_container_width=True,
                )
                shown_detections = detections

    show_candidates(image, shown_detections)

st.markdown(hidden_waldo_html(), unsafe_allow_html=True)
