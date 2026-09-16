"""Streamlit demo: upload a new Where's Waldo scene, watch the sliding-window
classifier search it live (heatmap building up, then NMS collapsing
candidates to a final box), or get YOLO's instant single-shot answer.

Run with: streamlit run app/main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patch_generation import PATCH_SIZE
from src.eval.box_utils import non_max_suppression
from src.models.patch_classifier import PatchClassifier
from src.models.sliding_window import DEFAULT_WINDOW_SIZES, windows_for_scale
from src.data.patch_dataset import to_tensor

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Finding Waldo", layout="wide")
st.title("🔍 Finding Waldo")
st.caption(
    "A from-scratch sliding-window CNN and a fine-tuned YOLO, trained on 19 scenes, "
    "searching a new Where's Waldo scene."
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


def draw_boxes(image: Image.Image, boxes_with_scores: list[tuple[tuple, float]], color: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for box, score in boxes_with_scores:
        draw.rectangle(box, outline=color, width=4)
        if score is not None:
            draw.text((box[0], max(0, box[1] - 14)), f"{score:.2f}", fill=color)
    return out


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

if sw_model is None and yolo_model is None:
    st.warning(
        "No trained models found in `models/`. Run notebooks 03 and 04 first to produce "
        "`sliding_window_classifier.pt` and `yolo_detector.pt`."
    )

uploaded = st.file_uploader("Upload a Where's Waldo scene", type=["jpg", "jpeg", "png"])

available_models = [name for name, m in [("Sliding-window CNN (from scratch)", sw_model), ("YOLO (fine-tuned)", yolo_model)] if m is not None]
model_choice = st.radio("Model", available_models, horizontal=True) if available_models else None

if uploaded is not None and model_choice is not None:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input scene")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Result")
        placeholder = st.empty()

        if model_choice.startswith("Sliding-window"):
            score_threshold = st.slider("Detection confidence threshold", 0.1, 0.95, 0.6, 0.05)
            if st.button("Search for Waldo"):
                detections = run_sliding_window_live(sw_model, image, score_threshold, placeholder)
                result = draw_boxes(image, detections, color="red")
                placeholder.image(result, caption=f"{len(detections)} detection(s) after NMS", use_container_width=True)
        else:
            conf = st.slider("Detection confidence threshold", 0.1, 0.95, 0.4, 0.05)
            if st.button("Search for Waldo"):
                yolo_result = yolo_model.predict(image, conf=conf, verbose=False)[0]
                detections = [
                    (tuple(b.xyxy[0].tolist()), b.conf.item()) for b in yolo_result.boxes
                ]
                result = draw_boxes(image, detections, color="red")
                placeholder.image(result, caption=f"{len(detections)} detection(s)", use_container_width=True)
