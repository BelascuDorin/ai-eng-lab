"""Read logs/agent_runs.jsonl and print one row per agent run.

This is the payoff of structured logs: a few lines of code answer "which runs were
slow / expensive / hit errors?" - questions print() output can't answer.

Run:  uv run log_summary.py           (last 20 runs)
      uv run log_summary.py 50        (last 50 runs)
"""

import json
import sys
from collections import defaultdict

from run_log import LOG_PATH


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    if not LOG_PATH.exists():
        print(f"No log yet at {LOG_PATH}. Run the agent first.")
        return

    # Group lines back into runs by run_id. dicts keep insertion order = chronological.
    runs: dict[str, list[dict]] = defaultdict(list)
    with LOG_PATH.open() as f:
        for line in f:
            event = json.loads(line)
            runs[event["run_id"]].append(event)

    print(f"{'run_id':<9} {'scenario':<22} {'end':<11} {'turns':>5} {'tools':>5} {'errs':>4} "
          f"{'tok in':>7} {'tok out':>7} {'cost $':>8} {'secs':>6}")
    for run_id, events in list(runs.items())[-limit:]:
        start = next(e for e in events if e["event"] == "run_start")
        end = next((e for e in events if e["event"] == "run_end"), None)
        scenario = start["metadata"].get("scenario", "(repl)")
        if end is None:  # process died mid-run: the log still tells you how far it got
            print(f"{run_id:<9} {scenario:<22} {'CRASHED?':<11} ({len(events)} events logged)")
            continue
        print(f"{run_id:<9} {scenario:<22} {end['reason']:<11} {end['turns']:>5} {end['tool_calls']:>5} "
              f"{end['tool_errors']:>4} {end['input_tokens']:>7} {end['output_tokens']:>7} "
              f"{end['cost_usd']:>8.4f} {end['latency_ms'] / 1000:>6.1f}")


if __name__ == "__main__":
    main()
