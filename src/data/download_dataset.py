"""One-time script: download the raw 'where's waldo' dataset from Roboflow
into data/raw/. Re-run any time to re-fetch (idempotent — Roboflow's client
skips re-downloading if the target folder already exists).

Requires ROBOFLOW_API_KEY in a .env file at the project root (gitignored).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ["ROBOFLOW_API_KEY"]

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("ml-9naud").project("where-s-waldo-vugud")
    version = project.version(1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = version.download("voc", location=str(RAW_DATA_DIR / "wheres-waldo-voc"))
    print(f"Downloaded dataset to: {dataset.location}")


if __name__ == "__main__":
    main()
