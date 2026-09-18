"""Colab-only plumbing: move work between Google Drive and Colab's local disk
without ever depending on Drive for many small files.

Drive is a network filesystem with two properties that bit us:
  * per-file latency is high, so a training loop opening thousands of small
    patch images per epoch takes hours instead of minutes;
  * writes are lazy -- copying thousands of small files "finishes" locally
    long before Drive has them, so a notebook started in a fresh runtime can
    see a partial copy.

So the rules here are: work only on Colab's local disk; move artifacts between
notebooks as *single zip files* (one big sequential transfer, and a truncated
zip is detectable); flush Drive explicitly before declaring a sync done; and
verify every file a manifest references actually exists before training.

This module is self-contained (stdlib only, no `src.*` imports) because the
first notebook cell loads it straight from Drive by file path, before the
local copy of the project exists.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

LOCAL_ROOT = Path("/content/wheres-wally")
SYNC_DIRNAME = "_sync"
# Copied from Drive on every fresh runtime. Deliberately excludes data/,
# models/, runs/ and notebooks/: those either come back via _sync zips or
# aren't needed at runtime (Colab opens the notebook itself from Drive).
BASE_ITEMS = ("src", "requirements.txt", ".env")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


def setup_local_project(
    drive_root: Path | str, local_root: Path | str = LOCAL_ROOT, extra_items: Iterable[str] = ()
) -> Path:
    """Copy the code (plus `extra_items`, e.g. "Hey-Waldo") from Drive to local
    disk, then restore whatever earlier notebooks synced. Idempotent and safe
    to re-run: an item is only skipped if a completion marker says its copy
    finished, so a copy interrupted partway is redone rather than trusted.
    """
    drive_root, local_root = Path(drive_root), Path(local_root)
    done_dir = local_root / ".bootstrap_done"
    done_dir.mkdir(parents=True, exist_ok=True)

    for item in (*BASE_ITEMS, *extra_items):
        src = drive_root / item
        if not src.exists():
            print(f"(skipping {item}: not found on Drive)")
            continue

        dst = local_root / item
        marker = done_dir / item.replace("/", "__")
        if marker.exists() and dst.exists():
            continue

        if dst.exists():  # leftover from an interrupted copy
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        print(f"Copying {item} from Drive to local disk...")
        if src.is_dir():
            shutil.copytree(src, dst, ignore=_IGNORE)
        else:
            shutil.copy2(src, dst)
        marker.write_text("ok")

    _restore_synced_artifacts(drive_root, local_root, done_dir)
    return local_root


def _restore_synced_artifacts(drive_root: Path, local_root: Path, done_dir: Path) -> None:
    sync_dir = drive_root / SYNC_DIRNAME
    if not sync_dir.exists():
        return

    # Oldest first, so a later sync of the same path wins.
    for z in sorted(sync_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime):
        stat = z.stat()
        stamp = f"{int(stat.st_mtime)}:{stat.st_size}"
        marker = done_dir / f"restored__{z.name}"
        if marker.exists() and marker.read_text() == stamp:
            continue

        with tempfile.TemporaryDirectory() as tmp:
            local_zip = Path(tmp) / z.name
            shutil.copyfile(z, local_zip)  # one big sequential read from Drive
            try:
                with zipfile.ZipFile(local_zip) as zf:
                    bad = zf.testzip()
                    if bad:
                        raise zipfile.BadZipFile(f"corrupt entry {bad}")
                    for name in zf.namelist():
                        p = Path(name)
                        if p.is_absolute() or ".." in p.parts:
                            raise RuntimeError(f"Unsafe path in {z.name}: {name}")
                    zf.extractall(local_root)
            except zipfile.BadZipFile as exc:
                raise RuntimeError(
                    f"{z.name} on Drive is truncated or corrupt ({exc}). The notebook that produced it "
                    "probably didn't finish its final sync_to_drive cell. Re-run that notebook to its "
                    "end and wait for 'Flushed to Drive' before starting this one."
                ) from exc
        marker.write_text(stamp)
        print(f"Restored {z.name}")


def sync_to_drive(project_root: Path, drive_root: Path | None, *relative_paths: str) -> None:
    """Zip each of `relative_paths` (files or directories, relative to
    project_root) and copy the zips to Drive under `_sync/`, then flush Drive.
    The next notebook's `setup_local_project` restores them automatically.
    Also drops a plain copy of any `models/...` file on Drive at its normal path
    so you can download it directly. No-op outside Colab (drive_root is None).
    """
    if drive_root is None:
        return
    drive_root = Path(drive_root)
    _ensure_mounted(drive_root)

    sync_dir = drive_root / SYNC_DIRNAME
    sync_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for rel in relative_paths:
            src = project_root / rel
            if not src.exists():
                print(f"Skipping {rel} (doesn't exist locally)")
                continue

            zip_name = rel.replace("/", "__") + ".zip"
            local_zip = Path(tmp) / zip_name
            _zip_path(src, project_root, local_zip)
            shutil.copyfile(local_zip, sync_dir / zip_name)
            print(f"Synced {rel} -> {SYNC_DIRNAME}/{zip_name} ({local_zip.stat().st_size / 1e6:.1f} MB)")

            if src.is_file() and rel.startswith("models/"):
                (drive_root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, drive_root / rel)

    _flush_drive()


def verify_paths(paths: Iterable[str], label: str) -> None:
    """Fail fast, with a useful message, if any file a manifest lists is
    missing -- instead of crashing minutes into training inside a DataLoader
    worker.
    """
    paths = [str(p) for p in paths]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} files listed in {label} are missing on disk "
            f"(e.g. {missing[:3]}). This usually means the notebook that generated them didn't "
            "finish its final sync_to_drive cell (or Drive hadn't finished flushing) before this "
            "notebook started. Re-run that earlier notebook to its end, wait for 'Flushed to "
            "Drive', then restart this one."
        )
    print(f"{label}: all {len(paths)} referenced files present")


def _zip_path(src: Path, project_root: Path, out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        if src.is_file():
            zf.write(src, src.relative_to(project_root).as_posix())
        else:
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(project_root).as_posix())


def _ensure_mounted(drive_root: Path) -> None:
    if drive_root.exists():
        return
    from google.colab import drive  # only reachable in Colab

    drive.mount("/content/drive")


def _flush_drive() -> None:
    try:
        from google.colab import drive
    except ImportError:
        return
    drive.flush_and_unmount()
    print("Flushed to Drive -- safe to start the next notebook.")
