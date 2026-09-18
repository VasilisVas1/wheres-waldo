# Running the project on Kaggle

The five notebooks in `notebooks/` are **the same files locally and on Kaggle** — they contain no
environment-specific code. Everything Kaggle-specific lives in this folder:

| File | Role |
|---|---|
| `build_package.py` | Builds the upload package from your **committed** project. Refuses to build if code/notebooks have uncommitted changes, and records every file's SHA-256. |
| `run_pipeline.py` | Runs on Kaggle. Finds the package, **verifies the checksums**, installs the few missing packages, executes notebooks 01→05 with papermill, and collects results. |
| `kaggle_runner.ipynb` | The only notebook you import into Kaggle; it just calls `run_pipeline.py`. |

So "what ran on Kaggle" is provably "commit X of this folder": the runner prints the commit and aborts
if any file differs from the build.

## One-time setup

1. **Kaggle account with phone verification.** GPU and Internet are both locked until the account is
   phone-verified (Settings → Phone verification).
2. **Build the package** (from the project root, with a clean `git status` for code/notebooks):
   ```bash
   python kaggle/build_package.py
   ```
   This writes `dist/wheres-waldo-project.zip`: code + notebooks + the raw Roboflow data + the parts of
   Hey-Waldo the pipeline uses. No API keys or secrets are included.
3. **Upload it as a dataset:** kaggle.com → *Datasets* → *New Dataset* → drop in the zip → title it
   `wheres-waldo-project` → visibility **Private** → *Create*. (Kaggle extracts uploaded zips itself.)
4. **Create the notebook:** kaggle.com → *Code* → *New Notebook* → *File* → *Import Notebook* → upload
   `kaggle/kaggle_runner.ipynb`.
5. In the notebook's right sidebar:
   - **Session options → Accelerator: `GPU T4 x2`.** Do **not** pick P100 — Kaggle's PyTorch has no
     kernels for it (the runner detects this and stops with a clear message).
   - **Session options → Internet: On.**
   - **Add Input** → *Your Datasets* → `wheres-waldo-project`.

## Run

**Save Version → Save & Run All (Commit).** This runs the whole thing on Kaggle's servers; you can close
the tab and laptop. Watch progress under *Save Version → the running version → Logs* if you like.

Budget: Kaggle allows ~30 GPU-hours/week and caps a GPU session at 9 hours. My *estimate* for the full
pipeline is roughly 1.5–3 hours (the sliding-window inference in notebook 03 is CPU-bound, and the whole
session counts against GPU time). That is an estimate, not a measurement.

## Getting results back

Open the finished version → **Output** tab → download. You get `results/`:

```
results/
  notebooks/           the five executed notebooks (with outputs)
  models/              sliding_window_classifier.pt, yolo_detector.pt
  data_processed/      cross-validation results, threshold sweeps, manifests (*.csv)
  runs/                YOLO training curves per fold (results.csv)
  PACKAGE_INFO.json    which commit ran
  pipeline_status.json per-notebook status and timings
```

Copy `models/*.pt` → `models/`, `notebooks/*.ipynb` → `notebooks/`, `data_processed/*.csv` →
`data/processed/`, then `git commit` the results. The Streamlit app (`streamlit run app/main.py`) then
uses the new models.

## After changing code or notebooks

1. Commit. 2. `python kaggle/build_package.py`. 3. Open your dataset on Kaggle → *New Version* → upload
the new zip. 4. Open the notebook and make sure the attached input shows the newest dataset version, then
run again.

## If something goes wrong

| Message | Meaning / fix |
|---|---|
| `No GPU is attached` | Session options → Accelerator → GPU T4 x2 |
| `can't run the installed PyTorch ... P100` | Session options → Accelerator → GPU T4 x2 (not P100) |
| `pip install failed. Is Internet turned on?` | Session options → Internet → On (needs phone verification) |
| `Project package not found under /kaggle/input` | Add Input → your dataset; check what's attached (the message lists it) |
| `package file(s) are missing or differ` | Stale or partial upload: rebuild the package and upload a new dataset version |
| `PIPELINE FAILED at 0X_...` | Notebook 0X errored. It doesn't abort the save: `results/notebooks/0X_....ipynb` contains everything up to the failing cell, with the traceback |

## What has and hasn't been verified

Verified locally against a simulated Kaggle layout: package build and integrity checking, package
discovery (extracted and zipped layouts), project assembly, papermill execution of notebooks 01–02 with
`cwd=notebooks/`, result collection, and the failure paths. **Not verified on Kaggle itself** (I can't
run there): GPU behaviour, the pip installs, and the runtime of notebooks 03–04. The runner is written to
fail fast and loudly on the things that most plausibly differ.
