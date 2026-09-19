# Finding Waldo — CV Object Localization + Live Demo

A from-scratch computer vision project that finds Waldo (Wally) hidden inside dense,
cluttered "Where's Waldo?" scenes. Three detectors are built and compared:

1. **Sliding-window CNN classifier** (`src/models/patch_classifier.py`) — a binary
   "is this crop Waldo or not?" CNN, slid across a full scene at multiple scales to
   build a heatmap, then thresholded and reduced with hand-implemented non-max
   suppression. Built from scratch so the mechanics of detection (IoU, NMS, sliding
   windows) are visible in code, not hidden behind a framework call.
2. **YOLO11n on the whole scene** (`ultralytics`) — the scene shrunk to 640×640, as a
   framework-based baseline.
3. **YOLO11n on native-resolution tiles** (`src/data/tiling.py`) — the same model, but trained
   and run on overlapping 640×640 tiles cut from the full-size scans, optionally at several
   image scales. This is the best detector here.

All are wrapped in a Streamlit app that runs inference on a new scene.

## Current status, honestly

The tiled YOLO finds Waldo more often than not, but it is not a reliable finder. Measured with
scene-level 5-fold cross-validation (every scene scored by a model that never trained on it;
IoU ≥ 0.3; 21 Waldo boxes in 18 scenes that contain him):

| model | found | false positives | missed | precision | recall |
|---|---|---|---|---|---|
| Sliding-window CNN (from scratch) | 8 | 4,203 | 13 | 0.002 | 0.38 |
| YOLO11n, whole scene at 640 px | 2 | 22 | 19 | 0.083 | 0.095 |
| YOLO11n, native tiles | 6 | 0 | 15 | 1.00 | 0.29 |
| YOLO11n, native tiles, multi-scale | 7 | 14 | 14 | 0.33 | 0.33 |

The most useful readout is ranking — *is Waldo among the detector's top-k boxes?* — since a puzzle
has one Waldo. For the tiled YOLO with multi-scale scanning he is the **#1 box in 10 of 18 scenes,
in the top 3 in 13 (72%), and in the top 10 in 14**. The demo therefore shows the top few candidates.

Things to know before trusting any of this:

- Only 21 boxes / 18 scenes, so every number has wide error bars (re-running the whole-scene YOLO
  swung it from 4 hits to 2).
- The multi-scale variant was picked out of four on the same scenes, so it is somewhat optimistic.
- Three annotated "Waldos" are Waldo portraits on postcard stamps, not hidden figures.
- The demo's tiled model is trained on all 19 scenes, so there is nothing held out to score it on;
  the cross-validated numbers are the estimate of how it should behave.
- Scenes 2, 3 and 13 are never found at confidence ≥ 0.01, and I haven't worked out why.

**A data bug worth knowing about.** Roboflow's export flipped or rotated 8 of the 19 pages relative to
the Hey-Waldo scans. Mapping its boxes onto the scans by rescaling alone put the "ground truth" on the
wrong content for those scenes, which capped every native-resolution result until notebook 02 learned to
detect each page's orientation (and refuse to continue without a clear match). Fixing it took the tiled
YOLO from "Waldo in the top-10 for 4 of 18 scenes" to 10 of 18 with the same training recipe.
Full write-up: notebook 06.

## Project structure

```
notebooks/          six pipeline stages, each with markdown explaining *why*
  01_data_exploration.ipynb
  02_preprocessing.ipynb           (also: native-scan alignment, crop pool)
  03_train_sliding_window.ipynb
  04_train_yolo.ipynb              (whole-scene YOLO baseline)
  05_tiled_yolo.ipynb              (native-resolution tiles; the best detector)
  06_evaluation_error_analysis.ipynb
src/
  data/              dataset loading, patches, augmentation, native alignment, tiling
  models/            sliding-window CNN, IoU/NMS implementation
  eval/              detection metrics (precision/recall @ IoU, localization error)
tests/               box-mapping tests for the native-scan alignment
app/                 Streamlit demo app
run_all.py           runs the notebooks unattended (e.g. overnight)
data/
  raw/               downloaded source dataset (gitignored)
  processed/         generated patches / splits / results (gitignored)
Hey-Waldo/           second dataset, from Kaggle (gitignored; see Dataset)
models/              trained weights (gitignored)
```

## Dataset

[Roboflow "where's waldo"](https://universe.roboflow.com/ml-9naud/where-s-waldo-vugud) —
65 raw images, single class (`waldo`), bounding-box annotated, CC BY 4.0, exported in
Pascal VOC XML format. Notebook 01 found that most of these 65 images are actually
unrelated portrait closeups rather than genuine "hidden in a crowd" puzzle scenes; after
filtering to the genuine ones, **19 clean scenes** remain and are what the rest of the
pipeline is built on (`data/processed/clean_manifest.csv`). With so few scenes, splitting
happens via **scene-level 5-fold cross-validation** (notebook 02), not one fixed
train/val/test split — and always at the scene level, never patch level, to avoid leakage
between crops of the same image.

The same 19 scenes also come from a second source,
[Hey-Waldo](https://www.kaggle.com/datasets/residentmario/wheres-waldo) (place it at `Hey-Waldo/`).
Roboflow supplies the bounding-box annotations (in its own 640×640, sometimes flipped/rotated, images);
Hey-Waldo supplies the native-resolution scans and ~3,000 extra classification patches for the
sliding-window model. Notebook 02 aligns the two (see the data-bug note above).

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

Work through `notebooks/01_...` to `notebooks/06_...` in order — each stage depends
on artifacts produced by the previous one (raw data → patches/splits/aligned scans → trained
sliding-window model → whole-scene YOLO → tiled YOLO → evaluation).

### Running everything unattended

Training (notebooks 03–05) takes hours on a CPU (roughly 4–5 h for notebook 05 alone on a laptop), so run
the pipeline in the background:

```bash
python run_all.py --detach       # notebooks 01 -> 06 in order; stops at the first failure
python run_all.py --from 04      # resume from a notebook (e.g. after an interruption)
python run_all.py --only 05,06   # run just these notebooks
```

It keeps Windows from idle-sleeping while it runs (closing the lid can still sleep the machine, so
leave it open and plugged in). Progress is in `logs/status.json` and `logs/<notebook>.log`; each
notebook is saved in place with its outputs when it finishes. Notebooks 04 and 05 skip folds that
already finished training, so an interrupted run resumes.

## Running the demo app

```bash
streamlit run app/main.py
```

Pick "YOLO on native-resolution tiles" and upload a full-size scan (1,300 px+ wide; a small web image
gives Waldo too few pixels). It scans at three scales, outlines the top candidates with numbered boxes, and
shows an enlarged close-up of each one below the scene (the best guess largest), so you don't have to zoom
into the full image to find the box. Scanning takes several seconds on a CPU; after that, the confidence and
"how many candidates" sliders re-filter the same scan instantly.

## Tests

```bash
python -m pytest tests
```
