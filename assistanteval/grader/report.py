"""Grade every run in a folder and report the pass rate.

    uv run python -m assistanteval.grader.report runs/ --out grades/

Each run is graded against the task its run.json holds. Verdicts go to <out>/<run_id>/verdict.json, and every run
shares the answers file <out>/answers.json. The pass rate is pass / (pass + fail + fail_unsafe). Invalid runs are not
counted. Runs that need human annotation are pending: answer them with assistanteval.grader.annotate, then run this
again with --offline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from assistanteval import run_record
from assistanteval.grader import (
    GRADER_VERSION,
    AnswerStore,
    grade,
    ladder_from_env,
)

COLUMNS = (
    "pass",
    "fail",
    "fail_unsafe",
    "needs_human_annotation",
    "invalid",
    # A run whose reset kept state shared with earlier runs: graded, but not
    # a clean-slate run, so kept out of the pass rate.
    "not_isolated",
)
ANNOTATIONS = (
    "annotation_questions",
    "runs_with_annotations",
    "decided_runs_with_annotations",
)


def pass_rate(counts: Counter) -> float | None:
    scored = counts["pass"] + counts["fail"] + counts["fail_unsafe"]
    return round(counts["pass"] / scored, 3) if scored else None


def interval(counts: Counter, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval for the pass rate: with a handful of runs per task, the uncertainty is large."""
    n = counts["pass"] + counts["fail"] + counts["fail_unsafe"]
    if not n:
        return None
    p = counts["pass"] / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


TASK_ID = re.compile(r"[a-z0-9][a-z0-9-]*")  # a task file's name, never a path


def run_folders(root: Path) -> list[Path]:
    """Every run folder under root, at any depth."""
    return sorted({p.parent for p in root.rglob("run.json")})


def _row(counts: Counter, annotations: Counter) -> dict:
    """A report row's counts, annotation counts and pass rate."""
    return {
        "runs": counts.total(),
        **{k: counts[k] for k in COLUMNS},
        **{k: annotations[k] for k in ANNOTATIONS},
        "pass_rate": pass_rate(counts),
        "pass_rate_95": interval(counts),
    }


def _print(rows: list[dict], overall: list[dict], skipped: list[dict]) -> None:
    print(
        f"{'product':9} {'task':30} {'runs':>4} {'pass':>4} {'fail':>4} {'unsafe':>6} {'pending':>7} "
        f"{'invalid':>7} {'pass rate (95%)':>17} {'questions':>9} {'audit runs':>10} {'decided':>7}"
    )
    for r in rows + [{**o, "task_id": "ALL"} for o in overall]:
        rate = (
            "-"
            if r["pass_rate"] is None
            else "{:.0%} ({:.0%}-{:.0%})".format(r["pass_rate"], *r["pass_rate_95"])
        )
        print(
            f"{r['product']:9} {r['task_id']:30} {r['runs']:4} {r['pass']:4} {r['fail']:4} {r['fail_unsafe']:6} "
            f"{r['needs_human_annotation']:7} {r['invalid']:7} {rate:>17} "
            f"{r['annotation_questions']:9} {r['runs_with_annotations']:10} {r['decided_runs_with_annotations']:7}"
        )
    for r in rows:
        if r["not_isolated"]:
            print(
                f"{r['product']} {r['task_id']}: {r['not_isolated']} of {r['runs']} runs "
                "kept shared state from earlier runs; not in the pass rate "
                "(outcomes in grades/<run_id>/verdict.json)"
            )
    for s in skipped:
        print(f"skipped {s['run']}: {s['error']}")


def main(argv: list[str] | None = None, store=AnswerStore) -> int:
    """`store`: the answers file's class (see `grader.__main__.main`)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, help="folder that holds one folder per run")
    parser.add_argument("--out", type=Path, default=Path("grades"))
    parser.add_argument(
        "--offline", action="store_true", help="use saved answers only; no API calls"
    )
    args = parser.parse_args(argv)

    store = store(args.out / "answers.json")
    ladder = ladder_from_env(store, args.offline)
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    annotations: dict[tuple[str, str], Counter] = defaultdict(Counter)
    skipped = []  # runs that could not be read, so no product or task is known
    graded: set[str] = set()  # a second run with the same id would overwrite it
    try:
        for folder in run_folders(args.runs):
            try:
                record = run_record.load(folder)
                if not TASK_ID.fullmatch(record.task_id):
                    raise ValueError(f"task_id is not a task name: {record.task_id!r}")
                if record.run_id in graded:
                    raise ValueError(f"duplicate run id {record.run_id}")
            except (OSError, ValueError, KeyError, TypeError) as error:
                skipped.append(
                    {"run": str(folder.relative_to(args.runs)), "error": repr(error)}
                )
                continue
            try:
                verdict = grade(record, ladder)
            except Exception as error:  # noqa: BLE001 - one bad run must not stop the batch
                skipped.append(
                    {"run": str(folder.relative_to(args.runs)), "error": repr(error)}
                )
                continue
            graded.add(record.run_id)
            target = args.out / Path(record.run_id).name / "verdict.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(verdict, indent=1) + "\n")
            counts[(record.product, record.task_id)][
                # Invalid first: such a run counts nowhere either way.
                "not_isolated"
                if not verdict["isolated"] and verdict["outcome"] != "invalid"
                else verdict["outcome"]
            ] += 1
            pending = len(verdict["annotation_queue"])
            annotations[(record.product, record.task_id)].update(
                {
                    "annotation_questions": pending,
                    "runs_with_annotations": bool(pending),
                    "decided_runs_with_annotations": bool(pending)
                    and verdict["outcome"] in {"pass", "fail", "fail_unsafe"},
                }
            )
    finally:
        store.save()  # keep every paid answer, even if a run fails to load

    if not counts and not skipped:
        # A mistyped path is an error, and must not overwrite a previous report.
        print(f"No run records (run.json) found under {args.runs}")
        return 1

    rows, totals = [], defaultdict(Counter)
    annotation_totals = defaultdict(Counter)
    for (product, task_id), c in sorted(counts.items()):
        tally = annotations[(product, task_id)]
        rows.append({"product": product, "task_id": task_id, **_row(c, tally)})
        totals[product] += c
        annotation_totals[product] += tally
    overall = [
        {"product": p, **_row(c, annotation_totals[p])}
        for p, c in sorted(totals.items())
    ]
    report = {
        "grader_version": GRADER_VERSION,
        "overall": overall,
        "by_task": rows,
        "skipped": skipped,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")

    _print(rows, overall, skipped)
    # Every run failing (a revoked key, a judge outage) is not a clean report.
    return 1 if skipped and not counts else 0


if __name__ == "__main__":
    raise SystemExit(main())
