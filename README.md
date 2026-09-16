# Finding Waldo — CV Object Localization + Live Demo

A from-scratch computer vision project that finds Waldo (Wally) hidden inside dense,
cluttered "Where's Waldo?" scenes. Two detectors are built and compared:

1. **Sliding-window CNN classifier** (`src/models/patch_classifier.py`) — a binary
   "is this crop Waldo or not?" CNN, slid across a full scene at multiple scales to
   build a heatmap, then thresholded and reduced with hand-implemented non-max
   suppression. Built from scratch so the mechanics of detection (IoU, NMS, sliding
   windows) are visible in code, not hidden behind a framework call.
2. **YOLO fine-tune** (`ultralytics`) — a modern one-stage detector fine-tuned on the
   same data, compared honestly against the sliding-window baseline.

Both are wrapped in a Streamlit app that runs inference on a new scene and animates
the search — from raw pixels to a bounding box.

## Project structure

```
notebooks/          five pipeline stages, each with markdown explaining *why*
  01_data_exploration.ipynb
  02_preprocessing.ipynb
  03_train_sliding_window.ipynb
  04_train_yolo.ipynb
  05_evaluation_error_analysis.ipynb
src/
  data/              dataset loading, tiling/patch generation, augmentation
  models/            sliding-window CNN, IoU/NMS implementation
  eval/              detection metrics (precision/recall @ IoU, mAP, localization error)
app/                 Streamlit demo app, loads the best trained model
data/
  raw/               downloaded source dataset (gitignored)
  processed/         generated patches / splits (gitignored)
```

## Dataset

[Roboflow "where's waldo"](https://universe.roboflow.com/ml-9naud/where-s-waldo-vugud) —
65 full Where's Waldo puzzle scenes, single class (`waldo`), bounding-box annotated,
CC BY 4.0. Exported in Pascal VOC XML format. Train/val/test split is done at the
**scene level** (never patch level) to avoid leakage between crops of the same image.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Running the notebooks

```bash
jupyter lab
```

Work through `notebooks/01_...` to `notebooks/05_...` in order — each stage depends
on artifacts produced by the previous one (raw data → patches/splits → trained
sliding-window model → trained YOLO model → evaluation).

## Running the demo app

```bash
streamlit run app/main.py
```
