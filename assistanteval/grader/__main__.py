"""Grade one run record.

    uv run python -m assistanteval.grader <run_folder> [--answers grades/answers.json] [--offline]

A run.json record holds its task, rendered at the run's start.

Writes verdict.json to grades/<run_id>/ unless --out is given. Judge answers are read from and saved to
grades/answers.json, the file report and annotate share, so a person's annotations apply here too.
--offline uses saved answers only and makes no API calls; this reproduces published verdicts from their saved answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assistanteval import run_record
from assistanteval.grader import AnswerStore, grade, ladder_from_env


def main(argv: list[str] | None = None, store=AnswerStore) -> int:
    """`store`: the answers file's class, for a host whose saved answers need
    reading another way."""
    parser = argparse.ArgumentParser(description="Grade one run record.")
    parser.add_argument("run", type=Path, help="run record folder")
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path("grades/answers.json"),
        help="saved judge answers to reuse and extend (default: grades/answers.json)",
    )
    parser.add_argument(
        "--offline", action="store_true", help="use saved answers only; no API calls"
    )
    parser.add_argument(
        "--out", type=Path, help="output folder (default: grades/<run_id>)"
    )
    args = parser.parse_args(argv)

    record = run_record.load(args.run)
    out = args.out or Path("grades") / Path(record.run_id).name
    store = store(args.answers)
    ladder = ladder_from_env(store, args.offline)
    try:
        verdict = grade(record, ladder)
    finally:
        store.save()  # keep every paid answer, even if grading stops

    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(
        json.dumps({k: verdict[k] for k in ("run_id", "outcome", "judging")}, indent=1)
    )
    return 0 if verdict["outcome"] in ("pass", "fail", "fail_unsafe") else 2


if __name__ == "__main__":
    sys.exit(main())
