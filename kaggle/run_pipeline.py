"""Run this project's five notebooks, *unmodified*, on Kaggle.

The notebooks in `notebooks/` contain no environment-specific code, so the exact same files run
locally and here. All the environment handling lives in this script: it locates the uploaded
project package, verifies it is intact, lays it out on Kaggle's writable disk, installs the few
packages Kaggle's image lacks, executes notebooks 01-05 in order with papermill, and collects the
deliverables (executed notebooks, trained models, result CSVs) into `<workspace>/results/`.

Self-contained on purpose (stdlib only until the packages are installed): the Kaggle driver
notebook loads it straight from the read-only input dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

NOTEBOOKS = (
    "01_data_exploration.ipynb",
    "02_preprocessing.ipynb",
    "03_train_sliding_window.ipynb",
    "04_train_yolo.ipynb",
    "05_evaluation_error_analysis.ipynb",
)

# Everything else the code imports (torch, torchvision, numpy, pandas, scikit-learn, matplotlib,
# pillow, ipykernel) ships with Kaggle's image. albumentations is pinned to >=2.0 because the code
# uses its 2.x API; Kaggle's preinstalled copy may be older.
PIP_PACKAGES = ("ultralytics>=8.3", "albumentations>=2.0", "lxml", "pyyaml", "papermill")

PACKAGE_INFO = "PACKAGE_INFO.json"
HEY_WALDO_SCALES = ("64", "128", "256")


class PreflightError(RuntimeError):
    """Something is wrong with the setup itself (GPU, package, internet) -- nothing worth saving."""


# --------------------------------------------------------------------------- environment checks


def check_environment(require_gpu: bool) -> None:
    print(f"Python {platform.python_version()} on {platform.platform()}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Kaggle's image always has torch
        raise PreflightError("PyTorch is not installed in this environment.") from exc

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        if require_gpu:
            raise PreflightError(
                "No GPU is attached. Open the notebook's Session options (right sidebar) and set "
                "Accelerator to 'GPU T4 x2', then run again. (Notebook 04, YOLO training, needs it.)"
            )
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name} (compute capability {cap[0]}.{cap[1]}), {torch.cuda.device_count()} device(s)")
    try:
        (torch.zeros(8, device="cuda") + 1).sum().item()
    except Exception as exc:
        raise PreflightError(
            f"This GPU ({name}) can't run the installed PyTorch ({exc}). Kaggle's current PyTorch has no "
            "kernels for P100 (Pascal) GPUs. In Session options choose Accelerator = 'GPU T4 x2'."
        ) from exc


def install_packages(packages=PIP_PACKAGES) -> None:
    print("Installing:", ", ".join(packages))
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise PreflightError(
            "pip install failed. Is Internet turned on? (Session options -> Internet -> On; Kaggle "
            f"requires a phone-verified account for that.)\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )

    import importlib
    import importlib.metadata as md

    for module, dist in (("ultralytics", "ultralytics"), ("albumentations", "albumentations"),
                         ("lxml", "lxml"), ("yaml", "pyyaml"), ("papermill", "papermill")):
        importlib.invalidate_caches()
        try:
            importlib.import_module(module)
        except Exception as exc:
            raise PreflightError(f"'{module}' installed but can't be imported: {exc}") from exc
        print(f"  {dist} {md.version(dist)}")
    if int(md.version("albumentations").split(".")[0]) < 2:
        raise PreflightError("albumentations >= 2.0 is required (the code uses its 2.x API).")


# ------------------------------------------------------------------------------ package handling


def _walk_limited(root: Path, max_depth: int):
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        yield Path(dirpath), dirnames, filenames


def find_package(input_root: Path, workspace: Path, max_depth: int = 4) -> Path:
    """Locate the uploaded project package under input_root (Kaggle extracts uploaded zips into
    /kaggle/input/<dataset-slug>/, but the exact nesting is not something to rely on)."""
    for dirpath, _dirs, files in _walk_limited(input_root, max_depth):
        if PACKAGE_INFO in files:
            return dirpath

    for dirpath, _dirs, files in _walk_limited(input_root, max_depth):
        for f in files:
            if f.endswith(".zip"):
                try:
                    with zipfile.ZipFile(dirpath / f) as zf:
                        if PACKAGE_INFO in zf.namelist():
                            target = workspace / "_unpacked"
                            shutil.rmtree(target, ignore_errors=True)
                            print(f"Extracting {dirpath / f} ...")
                            zf.extractall(target)
                            return target
                except zipfile.BadZipFile:
                    continue

    listing = []
    for dirpath, dirs, files in _walk_limited(input_root, 2):
        listing.append(f"  {dirpath}  ({len(dirs)} dirs, {len(files)} files)")
    raise PreflightError(
        f"Project package not found under {input_root} (looked for {PACKAGE_INFO}). Did you attach "
        "the dataset to this notebook (Add Input -> your dataset)? What is attached:\n" + "\n".join(listing[:20])
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_writable(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IWUSR | (stat.S_IXUSR if p.is_dir() else 0))
        except OSError:
            pass


def assemble_project(package_dir: Path, workspace: Path) -> tuple[Path, dict]:
    """Copy the package to a writable location, unpack Hey-Waldo, verify every file against the
    checksums recorded when the package was built from the committed project state."""
    info = json.loads((package_dir / PACKAGE_INFO).read_text(encoding="utf-8"))
    print(f"Package: git commit {info['git_commit'][:10]}, built {info['built_at']}")

    project = workspace / "wheres-waldo"
    shutil.rmtree(project, ignore_errors=True)
    shutil.copytree(package_dir, project)
    _make_writable(project)

    hey_zip = project / "Hey-Waldo.zip"
    if hey_zip.exists() and not (project / "Hey-Waldo").exists():
        print("Unpacking Hey-Waldo ...")
        with zipfile.ZipFile(hey_zip) as zf:
            zf.extractall(project)
        hey_zip.unlink()

    bad = []
    for rel, digest in info["files"].items():
        p = project / rel
        if not p.exists() or _sha256(p) != digest:
            bad.append(rel)
    if bad:
        raise PreflightError(
            f"{len(bad)} package file(s) are missing or differ from what was built (e.g. {bad[:3]}). "
            "The upload is incomplete or stale -- rebuild the package and upload it again."
        )
    print(f"Verified {len(info['files'])} code/notebook files against the package checksums.")

    required = [project / "data" / "raw" / "wheres-waldo-voc" / s for s in ("train", "valid", "test")]
    required += [project / "Hey-Waldo" / "original-images"]
    required += [project / "Hey-Waldo" / s / lbl for s in HEY_WALDO_SCALES for lbl in ("waldo", "notwaldo")]
    missing = [str(p.relative_to(project)) for p in required if not p.is_dir()]
    if missing:
        raise PreflightError(f"Package is missing required data folders: {missing}")
    n_orig = len(list((project / "Hey-Waldo" / "original-images").glob("*.jpg")))
    if n_orig != 19:
        raise PreflightError(f"Expected 19 Hey-Waldo original images, found {n_orig}.")
    return project, info


# --------------------------------------------------------------------------------- execution


def run_notebook(project: Path, name: str, executed_dir: Path, kernel_name: str) -> None:
    """Execute one notebook with papermill in a subprocess (no event-loop clashes with the driver
    notebook), cwd = notebooks/ exactly as when run locally, streaming its output live."""
    nb_dir = project / "notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "papermill", str(nb_dir / name), str(executed_dir / name),
        "--kernel", kernel_name, "--cwd", str(nb_dir),
        "--log-output", "--no-progress-bar", "--start-timeout", "300",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line[:400], flush=True)
        tail.append(line)
        tail = tail[-25:]
    if proc.wait() != 0:
        raise RuntimeError("\n".join(tail[-12:]))


def collect_results(project: Path, executed_dir: Path, results_dir: Path, info: dict | None) -> None:
    """Copy only the deliverables out (Kaggle saves everything in the working dir as notebook
    output, so the scratch project is deleted afterwards to keep the output small)."""
    if results_dir.exists():
        shutil.rmtree(results_dir)
    (results_dir / "notebooks").mkdir(parents=True)

    if executed_dir.exists():
        for nb in sorted(executed_dir.glob("*.ipynb")):
            shutil.copy2(nb, results_dir / "notebooks" / nb.name)
    for src_dir, pattern, dst_name in (
        (project / "models", "*.pt", "models"),
        (project / "data" / "processed", "*.csv", "data_processed"),
    ):
        if src_dir.exists():
            for f in src_dir.glob(pattern):
                (results_dir / dst_name).mkdir(exist_ok=True)
                shutil.copy2(f, results_dir / dst_name / f.name)
    runs = project / "runs"
    if runs.exists():
        for f in runs.glob("*/results.csv"):
            (results_dir / "runs" / f.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, results_dir / "runs" / f.parent.name / f.name)
    if info is not None:
        (results_dir / PACKAGE_INFO).write_text(json.dumps(info, indent=1), encoding="utf-8")


def run(
    input_root: str | Path = "/kaggle/input",
    workspace: str | Path = "/kaggle/working",
    kernel_name: str = "python3",
    notebooks: tuple[str, ...] = NOTEBOOKS,
    install: bool = True,
    require_gpu: bool = True,
    keep_workspace: bool = False,
) -> dict:
    input_root, workspace = Path(input_root), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_dir = workspace / "results"

    # Preflight: anything wrong here is a setup problem, so fail immediately and loudly.
    check_environment(require_gpu)
    package_dir = find_package(input_root, workspace)
    project, info = assemble_project(package_dir, workspace)
    if install:
        install_packages()

    executed_dir = project / "executed"
    status: dict = {
        "git_commit": info["git_commit"], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notebooks": {}, "ok": False,
    }
    try:
        for name in notebooks:
            t0 = time.time()
            print(f"\n{'=' * 78}\n>>> {name}\n{'=' * 78}", flush=True)
            try:
                run_notebook(project, name, executed_dir, kernel_name)
            except Exception as exc:
                status["notebooks"][name] = {"status": "FAILED", "seconds": round(time.time() - t0), "error": str(exc)}
                status["failed_at"] = name
                print(f"\n!!! {name} FAILED:\n{exc}", flush=True)
                break
            status["notebooks"][name] = {"status": "ok", "seconds": round(time.time() - t0)}
            print(f"<<< {name} finished in {status['notebooks'][name]['seconds']}s", flush=True)
        else:
            status["ok"] = True
    except BaseException:
        status["error"] = traceback.format_exc()
        raise
    finally:
        collect_results(project, executed_dir, results_dir, info)
        (results_dir / "pipeline_status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
        if not keep_workspace:
            shutil.rmtree(project, ignore_errors=True)
            shutil.rmtree(workspace / "_unpacked", ignore_errors=True)

    banner = "PIPELINE COMPLETE" if status["ok"] else f"PIPELINE FAILED at {status.get('failed_at')}"
    print(f"\n{'#' * 78}\n{banner} -- deliverables are in {results_dir}\n{'#' * 78}")
    return status
