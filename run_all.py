"""Run notebooks 01-06 in order, unattended (e.g. overnight).

    python run_all.py --detach     # start in the background and return immediately
    python run_all.py              # run in this terminal
    python run_all.py --from 04    # resume from a notebook, e.g. after an interruption
    python run_all.py --only 02,05 # run just these notebooks, in order

Each notebook is executed in place: its outputs are saved into the .ipynb when it finishes. Stops at
the first failure. Progress: logs/status.json and logs/<notebook>.log. Keeps Windows from
idle-sleeping while it runs (closing the lid can still sleep the machine).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
STATUS = LOG_DIR / "status.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_status(status: dict) -> None:
    STATUS.write_text(json.dumps(status, indent=1), encoding="utf-8")


def keep_awake() -> None:
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def detach() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != "--detach"]]
    kwargs = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:  # 0x01000000 = CREATE_BREAKAWAY_FROM_JOB, so it outlives whatever launched it
            proc = subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kwargs)
        except OSError:
            proc = subprocess.Popen(cmd, creationflags=flags, **kwargs)
    else:
        proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
    print(f"Started in the background (pid {proc.pid}). Progress: {STATUS}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", default="01", help="first notebook number to run (default 01)")
    parser.add_argument("--only", default="", help="comma-separated notebook numbers to run, e.g. 02,05")
    parser.add_argument("--detach", action="store_true", help="run in the background")
    args = parser.parse_args()
    if args.detach:
        detach()
        return 0

    only = {n.strip().zfill(2) for n in args.only.split(",") if n.strip()}
    notebooks = [
        p
        for p in sorted((ROOT / "notebooks").glob("0[1-6]_*.ipynb"))
        if p.name[:2] >= args.start and (not only or p.name[:2] in only)
    ]
    if not notebooks:
        print(f"No notebooks to run from '{args.start}'.")
        return 2

    LOG_DIR.mkdir(exist_ok=True)
    keep_awake()
    status = {"started": now(), "pid": os.getpid(), "notebooks": {}, "finished": False, "ok": False}
    write_status(status)

    for nb in notebooks:
        t0 = time.time()
        status["notebooks"][nb.name] = {"status": "running", "started": now()}
        write_status(status)
        log_path = LOG_DIR / f"{nb.stem}.log"
        with open(log_path, "w", encoding="utf-8") as log:
            rc = subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.kernel_name=python3", "--ExecutePreprocessor.timeout=-1", str(nb)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            ).returncode
        status["notebooks"][nb.name].update(
            status="ok" if rc == 0 else "FAILED", seconds=round(time.time() - t0), log=str(log_path.relative_to(ROOT))
        )
        write_status(status)
        if rc != 0:
            break
    else:
        status["ok"] = True

    status["finished"], status["ended"] = True, now()
    write_status(status)
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
