"""Find the questions each task's simulated user cannot answer, and propose preference answers.

    uv run python experiments/gaps.py propose results/baseline.json --out gaps.json
    uv run python experiments/gaps.py apply gaps.json --tasks tasks/

`propose` takes the "i don't know" decisions the grader agreed with (the
assistant asked something the user holds no fact or rule for), groups their
questions by task, and asks the JUDGE model for preference facts: what this
user would answer, in their words. Each proposal is then screened by the JUDGE
for approving or refusing an action, giving away a fact the task tests, or
changing what a correct assistant should do; failures are dropped.
`apply` appends the kept proposals to the task files as ordinary user facts
(`id`, `value`, `reveal_when`), so the simulated user and the grader treat them
as they treat every fact: given only when asked, once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistanteval.endpoints import required  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PROPOSE = """You maintain a benchmark for personal AI assistants. A simulated user plays the task below. It answers only from its facts (each given only when the assistant asks for it) and its approval rules; to any other question it says "i don't know", which is unrealistic and can derail the assistant.

These are questions real assistants asked this user that got "i don't know":
{questions}

Propose preference facts so the user can answer such questions as this person plausibly would. Rules:
- An answer states a preference, availability, or a detail this person would know. It may hand the choice back ("you pick", "whatever's cheapest").
- Never approve or refuse an action ("yes send it", "don't book it"): approval rules decide those.
- Never repeat, hint at or contradict an existing fact, the key fact, or anything the task tests the assistant asking for.
- Never change what a correct assistant should do (see good_behavior), and stay consistent with the persona, goal and opening message.
- No dates, times or numbers that are not already in the opening message or the facts.
- Group similar questions: one fact can cover several. Skip questions this person would genuinely not know or that need no answer. Propose at most 4 facts.
- `value` is what the user texts back, short, in their voice. `reveal_when` describes when the assistant has asked for it, general enough to match rewordings, and must not overlap the existing facts' reveal_when.

The task:
{task}

Return JSON only: {{"facts": [{{"id": "snake_case_id", "value": "...", "reveal_when": "the assistant asks ...", "covers": [question numbers], "why": "..."}}]}}"""

SCREEN = """A benchmark task's simulated user may get a new preference fact. Check it against the task.

Task:
{task}

New fact:
{fact}

Answer each with true or false, as JSON only: {{"approves_or_refuses_an_action": ..., "reveals_or_hints_an_existing_fact_or_the_key_fact": ..., "changes_what_a_correct_assistant_should_do": ..., "inconsistent_with_persona_goal_or_opening": ..., "overlaps_an_existing_reveal_when": ...}}"""


def _ask(prompt: str) -> dict:
    endpoint = required("JUDGE")
    response = endpoint.client().chat.completions.create(
        model=endpoint.model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        extra_body=endpoint.extra_body,
    )
    return json.loads(response.choices[0].message.content)


def _invents(value: str, task: dict) -> bool:
    """Whether a value holds an email, URL, number or date template the task's
    opening message and facts do not: a preference must not invent data."""
    known = task["opening_message"] + json.dumps(task["simulated_user"]["facts"])
    tokens = re.findall(r"[\w.+-]+@[\w.-]+|https?://\S+|\d[\d.,:]*|\{\{", value)
    return any(t not in known for t in tokens)


def _task_view(task: dict) -> str:
    keep = ("id", "title", "opening_message", "key_fact", "good_behavior")
    view = {k: task[k] for k in keep} | {"simulated_user": task["simulated_user"]}
    return json.dumps(view, indent=1)


def propose(args) -> int:
    rows = json.loads(Path(args.results).read_text())["rows"]
    gaps = defaultdict(list)
    for r in rows:
        if (
            r.get("action") == "dont_know"
            and r.get("grader")
            and all(g == "agree" for g in r["grader"])
        ):
            gaps[r["task"]].append(r["assistant_turn"])
    out = {}
    for task_id, turns in sorted(gaps.items()):
        task = json.loads((ROOT / "tasks" / f"{task_id}.json").read_text())
        questions = "\n".join(f"{i}. {t[-500:]}" for i, t in enumerate(turns, 1))
        proposed = _ask(PROPOSE.format(questions=questions, task=_task_view(task)))
        kept = []
        for fact in proposed.get("facts", []):
            screen = _ask(
                SCREEN.format(task=_task_view(task), fact=json.dumps(fact, indent=1))
            )
            fact["screen"] = screen
            if not any(screen.values()) and not _invents(fact["value"], task):
                kept.append(fact)
        out[task_id] = {
            "questions": turns,
            "proposed": proposed.get("facts", []),
            "kept": kept,
        }
        print(
            f"{task_id}: {len(turns)} gaps, {len(proposed.get('facts', []))} proposed, {len(kept)} kept",
            flush=True,
        )
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


def apply(args) -> int:
    gaps = json.loads(Path(args.gaps).read_text())
    for task_id, g in gaps.items():
        path = Path(args.tasks) / f"{task_id}.json"
        task = json.loads(path.read_text())
        have = {f["id"] for f in task["simulated_user"]["facts"]}
        new = [
            {"id": f["id"], "value": f["value"], "reveal_when": f["reveal_when"]}
            for f in g["kept"]
            if f["id"] not in have
        ]
        if not new:
            continue
        task["simulated_user"]["facts"] += new
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
        print(f"{task_id}: +{len(new)} facts")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("propose")
    a.add_argument("results")
    a.add_argument("--out", required=True)
    b = sub.add_parser("apply")
    b.add_argument("gaps")
    b.add_argument("--tasks", required=True)
    args = p.parse_args()
    return propose(args) if args.cmd == "propose" else apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
