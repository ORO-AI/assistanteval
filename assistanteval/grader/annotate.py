"""Answer the questions the judges could not settle.

    uv run python -m assistanteval.grader.annotate grades/*/verdict.json --answers grades/answers.json

Shows each queued question with its context and saves your answer as a human answer. Grade again with the same
answers file to use them. A human answer overrides both judges.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from assistanteval.grader.judges import HUMAN, AnswerStore, Ladder, Question


def ask(q: Question) -> bool | str | None:
    """The human's answer, or None to skip."""
    print(f"\n{'=' * 72}\n{q.text}")
    for name, value in q.context.items():
        shown = value if isinstance(value, str) else json.dumps(value, indent=1)
        print(f"\n[{name}]\n{shown}")
    if q.kind == "yes_no":
        reply = ""
        while reply not in ("y", "n", "s"):
            reply = input("\nyes, no, or skip [y/n/s]: ").strip().lower()[:1]
        return None if reply == "s" else reply == "y"
    options = list(q.criteria)
    for i, option in enumerate(options, 1):
        # The name tells apart options whose descriptions match, such as two
        # calendar.update requests.
        print(f"  {i}. {option}: {q.criteria[option]}")
    reply = input("\nnumber, or Enter to skip: ").strip()
    return (
        options[int(reply) - 1]
        if reply.isdigit() and 0 < int(reply) <= len(options)
        else None
    )


def _stale(store: AnswerStore, q: Question, value, decided_by: str) -> bool:
    """Whether the saved answers now decide `q` differently from the verdict.

    A recorded question has no `user_answer`. A reading the checker decided was
    either a re-check or a tie-break for the simulated user; either way, its
    value is the answer to replay it with."""
    if decided_by == "checker":
        q = replace(q, user_answer=value)
    current = Ladder(store).decide_one(q)
    return (current.value, current.decided_by) != (value, decided_by)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("verdicts", type=Path, nargs="+", help="verdict.json files")
    parser.add_argument(
        "--answers", type=Path, required=True, help="answers file to add to"
    )
    args = parser.parse_args(argv)

    store = AnswerStore(args.answers)
    for path in args.verdicts:
        changed = False
        queue = json.loads(path.read_text())["annotation_queue"]
        # Raw questions first: a causal answer is bound to their exact judgments.
        for item in sorted(
            queue, key=lambda i: i["question_id"] == "result_independent_of_simulator"
        ):
            q = Question(**item["question"])
            if q.id == "result_independent_of_simulator":
                for reading in q.context["reading_judgments"]:
                    changed |= _stale(
                        store,
                        Question(**reading["question"]),
                        reading["answer"],
                        reading["decided_by"],
                    )
                if changed:
                    print(
                        f"{path}: answers changed; regrade before causal adjudication"
                    )
                    continue
            else:
                changed |= _stale(store, q, item["value"], item["decided_by"])
            if store.get(HUMAN, q) is None and (answer := ask(q)) is not None:
                store.put(HUMAN, HUMAN, q, answer)
                store.save()
                changed = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
