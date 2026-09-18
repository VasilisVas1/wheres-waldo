"""Colab-only helper: sync generated outputs back to Drive at the end of a
notebook.

Google Drive is mounted as a network filesystem (FUSE) in Colab. That's fine
for a handful of large sequential reads, but our training loops open
thousands of small patch/image files individually, every epoch — at Drive's
per-file latency that turns minutes of real work into hours. The fix used
throughout these notebooks: the Colab bootstrap cell copies the whole
project to Colab's local disk (`/content/wheres-wally`) once per session and
everything runs against that local copy; this helper copies the *outputs*
(not the whole project) back to Drive at the end, as one bulk copy rather
than per-file-per-epoch access, so they persist across sessions and the
next notebook (possibly a fresh runtime) can pick them up.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def sync_to_drive(project_root: Path, drive_root: Path | None, *relative_dirs: str) -> None:
    """Copy each of `relative_dirs` (paths relative to project_root) from the
    local working copy back to the Drive-mounted project. No-op outside
    Colab (drive_root is None when running locally).
    """
    if drive_root is None:
        return

    for rel in relative_dirs:
        src = project_root / rel
        if not src.exists():
            print(f"Skipping sync for {rel} (doesn't exist locally)")
            continue
        dst = drive_root / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        print(f"Synced {rel} -> {dst}")
