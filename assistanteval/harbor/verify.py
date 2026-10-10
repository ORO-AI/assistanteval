"""Grade a Harbor trial's run record and write its reward.

    assistanteval harbor-verify <run record> <verifier logs dir>

Writes ``verdict.json`` and ``reward.json``: ``reward`` is 1 for a pass and 0
otherwise, with one 0/1 metric per outcome. Leave ``invalid`` and
``needs_human_annotation`` trials out of a pass rate, as the report does. No
run record (the trial was not run with AssistantEval's user agent) is invalid.
With no JUDGE role set, it grades with saved answers only, and the questions
code cannot settle wait for human annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assistanteval import endpoints, run_record
from assistanteval.grader import AnswerStore, grade, ladder_from_env

OUTCOMES = ("pass", "fail", "fail_unsafe", "needs_human_annotation", "invalid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.record / "run.json").is_file():
        store = AnswerStore(args.out / "answers.json")
        offline = endpoints.endpoint("JUDGE") is None
        verdict = grade(run_record.load(args.record), ladder_from_env(store, offline))
        store.save()
    else:
        verdict = {"outcome": "invalid", "detail": f"no run record at {args.record}"}
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    rewards = {o: int(verdict["outcome"] == o) for o in OUTCOMES}
    (args.out / "reward.json").write_text(
        json.dumps({"reward": rewards["pass"], **rewards})
    )
    print(verdict["outcome"])
    return 0
