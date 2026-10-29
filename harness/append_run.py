#!/usr/bin/env python3
"""Append a completed real_task_harness run to leaderboard.json.

Usage: append_run.py <run.json> <model-name> <engine> <thinking on|off> [note]

The harness prints a JSON array of per-task results after its progress lines; this
reads that array and maps it onto the leaderboard schema:
    {engine, mode, model, note, tasks: {<task>: {ans, elapsed, ok, replaced, think, why}}}

Existing entries with the same (model, mode) are replaced, not duplicated, so re-running
a model updates its row instead of stacking a second one.
"""
import json
import re
import sys
from pathlib import Path

LB = Path(__file__).resolve().parent.parent / "leaderboard.json"


def load_results(path):
    s = Path(path).read_text()
    # the array starts at the first line that is exactly "[" after the progress output
    m = re.search(r"^\[\s*$", s, re.M)
    if not m:
        raise SystemExit(f"no result array found in {path}")
    return json.loads(s[m.start():])


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    run_path, model, engine, thinking = sys.argv[1:5]
    note = sys.argv[5] if len(sys.argv) > 5 else ""

    results = load_results(run_path)
    tasks = {}
    for r in results:
        tasks[r["task"]] = {
            "ans": r.get("answer_tokens"),
            "elapsed": r.get("elapsed"),
            "ok": bool(r.get("passed")),
            "replaced": False,
            "think": r.get("thinking_tokens"),
            "why": "rubric-pass" if r.get("passed") else "rubric-fail",
        }
    entry = {"engine": engine, "mode": thinking, "model": model, "note": note, "tasks": tasks}

    board = json.loads(LB.read_text())
    before = len(board)
    board = [e for e in board
             if not (e.get("model") == model and e.get("mode") == thinking and e.get("engine") == engine)]
    board.append(entry)
    LB.write_text(json.dumps(board, indent=1) + "\n")

    passed = sum(1 for t in tasks.values() if t["ok"])
    print(f"  {model} [{engine}/{thinking}]: {passed}/4 tasks  ({before} -> {len(board)} entries)")


if __name__ == "__main__":
    main()
