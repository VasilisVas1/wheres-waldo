"""Build the Kaggle upload package from the project's *committed* state.

    python kaggle/build_package.py

The package is derived from your project folder, never hand-assembled, so what runs on Kaggle is
exactly what is committed here. It refuses to build if code or notebooks have uncommitted changes
(pass --allow-dirty to override, but then the recorded commit hash won't describe the contents).
Every code/notebook file's SHA-256 is recorded in PACKAGE_INFO.json; the runner re-checks them on
Kaggle before executing anything.

Output: dist/wheres-waldo-project.zip
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ("src", "notebooks", "kaggle", "requirements.txt", "README.md")
RAW_DIR = ROOT / "data" / "raw" / "wheres-waldo-voc"
HEY_WALDO_DIR = ROOT / "Hey-Waldo"
HEY_WALDO_PARTS = ("64", "128", "256", "original-images")  # the -gray / -bw variants are unused
OUT = ROOT / "dist" / "wheres-waldo-project.zip"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_hey_waldo_zip() -> bytes:
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # jpgs don't compress further
        for part in HEY_WALDO_PARTS:
            for f in sorted((HEY_WALDO_DIR / part).rglob("*")):
                if f.is_file():
                    zf.write(f, f"Hey-Waldo/{f.relative_to(HEY_WALDO_DIR).as_posix()}")
                    n += 1
    print(f"  Hey-Waldo.zip: {n} files, {buf.tell() / 1e6:.1f} MB")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    dirty = git("status", "--porcelain", "--", *CODE_PATHS).strip()
    if dirty and not args.allow_dirty:
        sys.exit(
            "Uncommitted changes in code/notebooks -- commit them first so the package matches a "
            f"commit exactly:\n{dirty}"
        )

    for path in (RAW_DIR, *(HEY_WALDO_DIR / p for p in HEY_WALDO_PARTS)):
        if not path.is_dir():
            sys.exit(f"Missing required data folder: {path}")

    commit = git("rev-parse", "HEAD").strip()
    tracked = [p for p in git("ls-files", "-z", "--", *CODE_PATHS).split("\0") if p]

    hashes = {rel: sha256_bytes((ROOT / rel).read_bytes()) for rel in tracked}
    info = {
        "git_commit": commit,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dirty_override": bool(dirty),
        "files": hashes,
    }

    OUT.parent.mkdir(exist_ok=True)
    print(f"Building {OUT.relative_to(ROOT)} from commit {commit[:10]}")
    hey = build_hey_waldo_zip()
    n_raw = 0
    def add_bytes(zf: zipfile.ZipFile, name: str, data: bytes, compress: int) -> None:
        # Explicit 0644: entries built from raw bytes otherwise get odd default permissions that some
        # extractors honour.
        zi = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
        zi.external_attr = 0o644 << 16
        zi.compress_type = compress
        zf.writestr(zi, data)

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        add_bytes(zf, "PACKAGE_INFO.json", json.dumps(info, indent=1).encode("utf-8"), zipfile.ZIP_DEFLATED)
        for rel in tracked:
            zf.write(ROOT / rel, rel)
        for f in sorted(RAW_DIR.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(ROOT).as_posix())
                n_raw += 1
        add_bytes(zf, "Hey-Waldo.zip", hey, zipfile.ZIP_STORED)

    print(f"  {len(tracked)} code/notebook files, {n_raw} raw-dataset files")
    print(f"Done: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, sha256 {sha256_bytes(OUT.read_bytes())[:16]}...)")


if __name__ == "__main__":
    main()
