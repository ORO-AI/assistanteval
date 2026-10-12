"""Replay the simulated user on recorded conversations, and check each decision as the grader would.

    uv run python experiments/probe.py sample --out probes.json [--per-task 7]
    uv run python experiments/probe.py run probes.json --out results/<arm>.json [--overlay tasks/]

`sample` picks published runs (downloaded to ~/ae-exp-data/published, see
README in this folder) stratified by task, oversampling runs whose user said
it did not know. `run` replays the current SimulatedUser on every point of
each sampled conversation where the user replied or the assistant stopped, with
the official judges, then asks the grader's re-read questions
(`grade._read_checks`) about each new decision. Judge answers are cached by
question, so two arms differ only where their questions differ.

With --overlay, user facts in that tasks folder that a published task lacks
(such as new preference answers) are added to it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

# AE_CODE runs another checkout's code (an earlier arm) with this harness.
sys.path.insert(
    0, os.environ.get("AE_CODE") or str(Path(__file__).resolve().parents[1])
)

from assistanteval.endpoints import endpoint, required  # noqa: E402
from assistanteval.grader.grade import _read_checks  # noqa: E402
from assistanteval.grader.judges import (  # noqa: E402
    AnswerStore,
    Asking,
    Ladder,
    judge_for,
)
from assistanteval.run_record import RunRecord, Turn  # noqa: E402
from assistanteval.simulated_user import SimulatedUser  # noqa: E402

DATA = Path.home() / "ae-exp-data"
DONT_KNOW = re.compile(r"\b(i don'?t know|dunno|no idea|not sure)\b", re.I)


# Recorded conversations ------------------------------------------------------


def published(run_id: str) -> tuple[dict, list[Turn], dict]:
    """A published run's task (as the simulated user sees it), turns and summary."""
    r = json.loads((DATA / "published" / f"{run_id}.json").read_text())
    t = r["task"]
    task = {
        "id": t["id"],
        "opening_message": t["opening_message"],
        "simulated_user": {
            "persona": t["persona"],
            "voice": t["voice"],
            "goal": t["goal"],
            "facts": t["user_facts"],
            "approvals": [
                {"if_not": "no, don't do that for now", **a} for a in t["approvals"]
            ],
            "otherwise": "",
        },
    }
    turns = [
        Turn(x["role"], x["text"], x["t"], x.get("nudge", False), x.get("edited_t"))
        for x in r["turns"]
    ]
    return task, turns, {"assistant": r["assistant"], "outcome": r["outcome"]}


def recorded(path: str) -> tuple[dict, list[Turn], dict]:
    """A full run record's task, turns and summary (our own runs)."""
    from assistanteval import run_record

    r = run_record.load(Path(path))
    return r.task, r.turns, {"assistant": r.product, "outcome": None}


def overlay(task: dict, tasks_dir: Path | None) -> dict:
    """The task with the facts its repo file has and it lacks."""
    if tasks_dir is None:
        return task
    path = tasks_dir / f"{task['id']}.json"
    if not path.is_file():
        return task
    have = {f["id"] for f in task["simulated_user"]["facts"]}
    extra = [
        f
        for f in json.loads(path.read_text())["simulated_user"]["facts"]
        if f["id"] not in have and "{{" not in json.dumps(f)
    ]
    user = {**task["simulated_user"], "facts": task["simulated_user"]["facts"] + extra}
    return {**task, "simulated_user": user}


def points(turns: list[Turn]) -> list[int]:
    """Indices of user replies (and nudges), then the end: where the user decided."""
    out = [i for i, t in enumerate(turns) if t.role == "user" and i > 0]
    if turns and turns[-1].role == "assistant":
        out.append(len(turns))
    return out


# Cached judges ----------------------------------------------------------------


class Cache:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.data = json.loads(path.read_text()) if path.is_file() else {}

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            tmp.replace(self.path)


class Cached:
    """A judge whose answers are cached by question key."""

    def __init__(self, judge, cache: Cache):
        self.judge, self.cache = judge, cache
        self.name, self.model = judge.name, judge.model
        self.identity = judge.identity

    def __call__(self, q):
        key = f"{self.identity}|{q.key()}"
        with self.cache.lock:
            if key in self.cache.data:
                return self.cache.data[key]
        answer = self.judge(q)
        with self.cache.lock:
            self.cache.data[key] = answer
        return answer


# Replay -----------------------------------------------------------------------


def replay(item: dict, reader, rules, ask, tasks_dir) -> list[dict]:
    task, turns, meta = (
        recorded(item["record"]) if "record" in item else published(item["run"])
    )
    item.setdefault("run", item.get("record"))
    task = overlay(task, tasks_dir)
    user = SimulatedUser(task, reader, None, rules=rules)
    rows = []
    for i in points(turns):
        seen = turns[:i]
        at = turns[i].at_ms if i < len(turns) else (seen[-1].at_ms or 0) + 1
        if i < len(turns) and turns[i].nudge:
            user.nudge(after_turn=i - 1, at_ms=at)
            continue
        try:
            d = user.reply(seen, at_ms=at).decision
        except Exception as exc:  # a judge failure: record it, keep going
            rows.append({"run": item["run"], "point": i, "error": type(exc).__name__})
            break
        sent = turns[i].text if i < len(turns) else None
        rows.append(
            {
                "run": item["run"],
                "task": task["id"],
                "assistant": meta["assistant"],
                "point": i,
                "saw": d["saw"],
                "action": d["action"],
                "fact_ids": d["fact_ids"],
                "operation": d["operation"],
                "text": d["text"],
                "recorded_reply": sent,
                "recorded_dont_know": bool(sent and DONT_KNOW.search(sent)),
                "assistant_turn": seen[-1].text[-600:] if seen else "",
            }
        )
    # The grader's re-read of each decision, on the conversation it was made in.
    record = RunRecord(
        item["run"],
        task["id"],
        meta["assistant"],
        turns,
        [],
        {},
        simulated_user={"decisions": user.decisions},
        task=task,
    )
    model = SimpleNamespace(
        simulated_user=SimpleNamespace(
            facts=[SimpleNamespace(**f) for f in task["simulated_user"]["facts"]],
            approvals=[
                SimpleNamespace(**{"only_if": None, **a})
                for a in task["simulated_user"]["approvals"]
            ],
        )
    )
    checks = {
        j: [
            (q, confirms, ask(q))
            for q, confirms, _ in _read_checks(record, model, d, j)
        ]
        for j, d in enumerate(user.decisions)
    }
    decided = [d for d in user.decisions]
    by_turn = {}
    for j, d in enumerate(decided):
        results = []
        unresolved = []
        for q, confirms, fut in checks[j]:
            v = fut.result().value
            results.append(
                "unsure" if v is None else ("agree" if v is confirms else "disagree")
            )
            if v is not confirms:
                unresolved.append(
                    {
                        "id": q.id,
                        "kind": q.kind,
                        "text": q.text,
                        "context": q.context,
                        "criteria": q.criteria,
                        "rungs": list(q.rungs),
                        "confirms": confirms,
                    }
                )
        by_turn[d["after_turn"]] = (results, unresolved)
    for row in rows:
        row["grader"], row["unresolved"] = by_turn.get(row["point"] - 1, ([], []))
    return rows


def run(args) -> int:
    items = json.loads(Path(args.probes).read_text())
    cache = Cache(DATA / "judge-cache.json")
    reader = Cached(judge_for(endpoint("READER")), cache)
    rules = Cached(judge_for(required("JUDGE")), cache)
    store = AnswerStore(DATA / "grader-answers.json")
    checker = endpoint("CHECKER")  # only code with a checker rung takes one
    extra = {"checker": Cached(judge_for(checker), cache)} if checker else {}
    ladder = Ladder(store, reader=reader, judge=rules, **extra)
    tasks_dir = Path(args.overlay) if args.overlay else None
    rows = []
    with Asking(ladder) as ask, ThreadPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(replay, it, reader, rules, ask, tasks_dir) for it in items
        ]
        for n, f in enumerate(futures, 1):
            try:
                rows += f.result()
            except Exception as exc:  # noqa: BLE001
                rows.append({"error": f"{type(exc).__name__}: {exc}"[:200]})
            if n % 20 == 0:
                cache.save()
                store.save()
                print(f"{n}/{len(items)} runs", flush=True)
    cache.save()
    store.save()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary(rows)}, indent=1))
    print(json.dumps(summary(rows), indent=1))
    return 0


def summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    decided = [r for r in ok if r.get("action")]
    dk = [r for r in decided if r["action"] == "dont_know"]
    disputed = [r for r in decided if any(g != "agree" for g in r.get("grader", []))]
    by = collections.Counter(r["saw"].split(":")[0] for r in dk)
    return {
        "decisions": len(decided),
        "errors": len(rows) - len(ok),
        "dont_know": len(dk),
        "dont_know_rate": round(len(dk) / max(1, len(decided)), 4),
        "dont_know_by_reading": dict(by),
        "disputed": len(disputed),
        "disputed_rate": round(len(disputed) / max(1, len(decided)), 4),
        "disputed_dont_know": sum(1 for r in disputed if r["action"] == "dont_know"),
        "actions": dict(collections.Counter(r["action"] for r in decided)),
    }


def sample(args) -> int:
    rng = random.Random(args.seed)
    exclude = set()
    for path in args.exclude or []:
        exclude |= {it["run"] for it in json.loads(Path(path).read_text())}
    by_task = collections.defaultdict(lambda: ([], []))
    for f in sorted((DATA / "published").glob("*.json")):
        if f.stem in exclude:
            continue
        r = json.loads(f.read_text())
        if not r.get("turns"):
            continue
        dk = any(
            t["role"] == "user" and i > 0 and DONT_KNOW.search(t["text"])
            for i, t in enumerate(r["turns"])
        )
        by_task[r["task_id"]][0 if dk else 1].append(r["id"])
    items = []
    for task, (dk, other) in sorted(by_task.items()):
        rng.shuffle(dk)
        rng.shuffle(other)
        take = dk[: args.per_task // 2]
        take += other[: args.per_task - len(take)]
        items += [{"run": run_id, "task": task} for run_id in take]
    Path(args.out).write_text(json.dumps(items, indent=1))
    print(f"{len(items)} runs from {len(by_task)} tasks")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--per-task", type=int, default=7)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--exclude", nargs="*", help="probe files whose runs to leave out")
    r = sub.add_parser("run")
    r.add_argument("probes")
    r.add_argument("--out", required=True)
    r.add_argument("--overlay")
    r.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    return sample(args) if args.cmd == "sample" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
