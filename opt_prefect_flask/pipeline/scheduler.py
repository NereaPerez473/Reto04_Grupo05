"""
pipeline/scheduler.py
=====================
Advances a 24-hour sliding window across the dataset and runs the full
pipeline on each window every SCHEDULER_INTERVAL seconds.

State is persisted in $DATA_DIR/pipeline_state.json so restarts resume
from where they left off.

Environment variables:
  APP_DIR             – project root  (default: parent of this file)
  SCHEDULER_INTERVAL  – seconds between runs (default: 120)
  WINDOW_HOURS        – hours per window (default: 24, must match T_HOURS)
  MAX_DATA_HOURS      – total hours in dataset before wrapping (default: 8736)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR    = Path(os.environ.get("APP_DIR", Path(__file__).resolve().parent.parent))
STATE_FILE  = BASE_DIR / "data" / "pipeline_state.json"
INTERVAL    = int(os.environ.get("SCHEDULER_INTERVAL", "120"))
WINDOW_H    = int(os.environ.get("WINDOW_HOURS", "24"))
MAX_HOURS   = int(os.environ.get("MAX_DATA_HOURS", "8736"))  # ~1 year


def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"start_hour": 0, "runs_completed": 0, "last_run": None,
            "last_start_hour": 0, "last_elapsed_s": None}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main() -> None:
    sys.path.insert(0, str(BASE_DIR))
    from pipeline.flow import pipeline_microred 

    print(f"[Scheduler] interval={INTERVAL}s  window={WINDOW_H}h  max_data={MAX_HOURS}h")
    print(f"[Scheduler] state file: {STATE_FILE}")

    while True:
        state = _load_state()
        start_hour = state["start_hour"]
        run_n = state["runs_completed"] + 1

        print(f"\n[Scheduler] === Run #{run_n}  window h{start_hour}–h{start_hour + WINDOW_H} ===")
        t0 = time.time()
        try:
            pipeline_microred(start_hour=start_hour)
            elapsed = time.time() - t0

            next_hour = (start_hour + WINDOW_H) % MAX_HOURS
            state.update({
                "start_hour":      next_hour,
                "last_start_hour": start_hour,
                "runs_completed":  run_n,
                "last_run":        datetime.now(timezone.utc).isoformat(),
                "last_elapsed_s":  round(elapsed, 1),
                "last_error":      None,
            })
            _save_state(state)
            print(f"[Scheduler] Done in {elapsed:.1f}s. Next window: h{next_hour}–h{next_hour + WINDOW_H}")

        except AssertionError:
            # Window exceeds available data — wrap around to hour 0
            print(f"[Scheduler] Window h{start_hour} exceeds data length — wrapping to h0")
            state["start_hour"] = 0
            state["last_error"] = "wrap-around: exceeded data length"
            _save_state(state)

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[Scheduler] ERROR after {elapsed:.1f}s: {exc}")
            state["last_error"] = str(exc)
            _save_state(state)

        print(f"[Scheduler] Sleeping {INTERVAL}s ...")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
