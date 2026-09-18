# Running this project on Google Colab

Heavy computation runs on Colab, not locally. In practice that means notebook 04 (YOLO training) on
Colab's free GPU; everything else is light enough for a plain CPU runtime.

## How data moves (read this once — it explains the rules below)

Google Drive is a network filesystem. It is fine for a few big files, but:

- opening thousands of small files (what training does every epoch) is extremely slow, and
- writes are *lazy* — copying thousands of small files "finishes" long before Drive actually has them.

So every notebook works **only on Colab's local disk** (`/content/wheres-wally`). Between notebooks,
results travel as **single zip files** in `My Drive/wheres-wally/_sync/`, and each notebook flushes
Drive before it finishes. The next notebook restores those zips automatically, and checks that every
file its manifests reference actually exists before it starts training.

## 1. One-time Drive setup

Create the folder **`My Drive/wheres-wally`** (exact name) and put in it:

- `src/`, `notebooks/`, `requirements.txt` — from `wheres-wally-code.zip` (replace older copies)
- `Hey-Waldo/` — your existing folder from Kaggle
- `.env` — one line: `ROBOFLOW_API_KEY=<your key>` (same key as your local `.env`)

You can delete any old `data/` folder in there — notebooks ignore it now. (A `models/` folder is
fine: trained models are dropped there for you to download.)

## 2. Run the notebooks in order

Open each from Drive: right-click → **Open with → Google Colaboratory**.

| Notebook | Runtime | Notes |
|---|---|---|
| `01_data_exploration` | **CPU** | Downloads the Roboflow dataset, cleans it to 19 scenes |
| `02_preprocessing` | **CPU** | Copies `Hey-Waldo/` locally (a few minutes, once), builds patches |
| `03_train_sliding_window` | **CPU** | Tiny CNN. Optional if you already have `sliding_window_classifier.pt` |
| `04_train_yolo` | **T4 GPU** | The only notebook that needs the GPU — saves your quota |
| `05_evaluation_error_analysis` | **CPU** | Final comparison |

Set the runtime with **Runtime → Change runtime type**. It doesn't carry over between notebooks.

**Run every notebook to its very last cell and wait for `Flushed to Drive — safe to start the next
notebook.`** before opening the next one. That last cell is what hands results to the next notebook;
if you start the next one early, it will fail its file check (on purpose) rather than train on
half-copied data.

If a notebook stops with *"N of M files listed in … are missing"* or *"… is truncated or corrupt"*,
the previous notebook didn't finish syncing: re-run that one to its end and try again.

## 3. Getting results back to your machine

When 03–05 are done, download from Drive into the same places in your local project:

- `models/sliding_window_classifier.pt`, `models/yolo_detector.pt`
- the executed `notebooks/*.ipynb` (Colab saves outputs into the Drive copy as you go)

Then `streamlit run app/main.py` locally uses the freshly trained models.
