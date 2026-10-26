"""Structured logging for the agent loop: one JSON object per line (JSONL).

Why JSONL instead of print()?
- print() is for a human watching live. JSONL is for *later*: you can grep it, load it
  into pandas, sum tokens per run, or replay a run as an eval case.
- One event per line means appending is cheap and a crash mid-run never corrupts
  earlier lines (unlike one big JSON array you'd have to rewrite).
- Every event carries a run_id, so you can group the lines back into one agent run.

Read it:  uv run log_summary.py        or   tail -f logs/agent_runs.jsonl | jq
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).parent / "logs" / "agent_runs.jsonl"


class RunLogger:
    """Writes events for ONE agent run (one call to run_agent) to the JSONL file."""

    def __init__(self, path: Path = LOG_PATH):
        self.run_id = uuid.uuid4().hex[:8]
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

    def log(self, event: str, **fields) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        # Open/append/close on every event: slower than keeping the file open, but each line
        # is on disk immediately, so you still have the log if the process crashes. Fine at this scale.
        # default=str turns anything json can't handle (e.g. SDK objects) into a string instead of crashing.
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
