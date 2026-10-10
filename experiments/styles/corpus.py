"""Build the style-overlay corpus: (task, persona, voice, recent conversation, fixed message) pairs.

    uv run python experiments/styles/corpus.py   # writes experiments/styles/corpus.json

Contexts are real conversations from the published runs (~/ae-exp-data/published).
A context is every turn before a real user reply (or the whole run, when it
ends on an assistant turn). The fixed message is what the simulated user's code
would send there: a fact's value, "yes, go ahead and <action>", the refusal
"no, don't do that for now", "i don't know", or an approval followed by a fact.
Each kind is placed where the recorded user's reply looks like that kind
(context_match="real_reply"); when no reply does, at a turn whose assistant
message asks something relevant (context_match="heuristic").
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import assistanteval.grader  # noqa: E402,F401  (import order: grader before simulated_user)
from assistanteval.simulated_user import FIXED, approval  # noqa: E402

DATA = Path.home() / "ae-exp-data" / "published"
OUT = Path(__file__).with_name("corpus.json")
TARGET = 120
SEED = 20261010

DONT_KNOW = re.compile(r"\b(i don'?t know|dunno|no idea|not sure)\b", re.I)
YES = re.compile(
    r"\b(yes|yeah|yep|yup|go ahead|sounds good|do it|send it|ok|okay|sure|perfect|looks good|that works|approved?)\b",
    re.I,
)
NO = re.compile(r"^\W*(no|nope|don'?t|do not|hold off|not yet|wait)\b", re.I)
OP_WORDS = {
    "marketplace.reply": ["reply", "send", "message"],
    "calendar.create": ["calendar", "event"],
    "calendar.update": ["calendar", "move", "reschedule"],
    "amazon.checkout": ["order", "checkout", "place", "pay"],
    "amazon.cart_add": ["cart"],
    "gmail.draft_update": ["draft", "save"],
    "gmail.send": ["send", "email"],
    "email.send": ["send", "email"],
    "courier.save": ["pickup", "courier", "book"],
    "flights.change": ["flight", "change", "switch"],
    "subscriptions.cancel": ["cancel"],
}
WORD = re.compile(r"[a-z0-9]+")
STOP = set(
    "the a an and or to of in on for is it i my me you your that this be with at as are was "
    "just so but not no yes if do don t s".split()
)


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1}


def overlap(fact_value: str, reply: str) -> float:
    fv = words(fact_value)
    return len(fv & words(reply)) / len(fv) if fv else 0.0


def op_score(op: str, assistant_text: str) -> int:
    low = assistant_text.lower()
    return sum(w in low for w in OP_WORDS.get(op, [op.split(".")[1]]))


def runs():
    for path in sorted(DATA.glob("*.json")):
        r = json.loads(path.read_text())
        if "turns" in r and "task" in r:
            yield r


def points(turns: list[dict]):
    """(index of the reply or len(turns), the real reply text or None)."""
    for i, t in enumerate(turns):
        if i > 0 and t["role"] == "user" and not t.get("nudge"):
            if turns[i - 1]["role"] == "assistant":
                yield i, t["text"]
    if turns and turns[-1]["role"] == "assistant":
        yield len(turns), None


def candidates():
    """Per task, per kind: candidate (run, point, op/fact, match) placements."""
    by_task: dict[str, dict] = {}
    for r in runs():
        task = r["task"]
        tid = task["id"]
        entry = by_task.setdefault(
            tid, {"task": task, "cands": collections.defaultdict(list)}
        )
        yes_ops = [a["operation"] for a in task["approvals"] if a["answer"] == "yes"]
        all_ops = [a["operation"] for a in task["approvals"]]
        for i, reply in points(r["turns"]):
            last = r["turns"][i - 1]["text"]
            asks = "?" in last
            base = {"run_id": r["id"], "point": i}
            cands = entry["cands"]
            if reply is not None:
                fact_hits = [
                    (overlap(f["value"], reply), f["id"]) for f in task["user_facts"]
                ]
                best = max(fact_hits, default=(0, None))
                is_dk = bool(DONT_KNOW.search(reply))
                is_no = bool(NO.search(reply))
                is_yes = bool(YES.search(reply)) and not is_no
                if is_dk:
                    cands["dont_know"].append({**base, "match": "real_reply"})
                if best[0] >= 0.3:
                    cands["reveal"].append(
                        {**base, "fact": best[1], "match": "real_reply"}
                    )
                if is_yes and yes_ops:
                    op = max(yes_ops, key=lambda o: op_score(o, last))
                    kind = "approve_reveal" if best[0] >= 0.3 else "approve"
                    cands[kind].append(
                        {
                            **base,
                            "op": op,
                            "fact": best[1] if best[0] >= 0.3 else None,
                            "match": "real_reply",
                        }
                    )
                if is_no and all_ops:
                    op = max(all_ops, key=lambda o: op_score(o, last))
                    cands["refuse"].append({**base, "op": op, "match": "real_reply"})
            cands["_any"].append(base)
            if asks:
                cands["_asks"].append(base)
    return by_task


def build() -> list[dict]:
    rng = random.Random(SEED)
    by_task = candidates()
    run_turns = {r["id"]: r["turns"] for r in runs()}
    plans = {}
    for tid, entry in by_task.items():
        task, cands = entry["task"], entry["cands"]
        facts = [f["id"] for f in task["user_facts"]]
        yes_ops = [a["operation"] for a in task["approvals"] if a["answer"] == "yes"]
        all_ops = [a["operation"] for a in task["approvals"]]
        feasible = ["dont_know"]
        if facts:
            feasible.append("reveal")
        if yes_ops:
            feasible.append("approve")
        if all_ops:
            feasible.append("refuse")
        if facts and yes_ops:
            feasible.append("approve_reveal")
        plans[tid] = feasible

    counts = collections.Counter()
    used = collections.defaultdict(set)
    corpus = []
    tids = sorted(plans)
    while len(corpus) < TARGET:
        rng.shuffle(tids)
        for tid in tids:
            if len(corpus) >= TARGET:
                break
            task, cands = by_task[tid]["task"], by_task[tid]["cands"]
            left = [k for k in plans[tid] if k not in used[tid]] or plans[tid]
            kind = min(left, key=lambda k: (counts[k], rng.random()))
            used[tid].add(kind)
            pool = [
                c
                for c in cands.get(kind, [])
                if (c["run_id"], c["point"]) not in used[tid]
            ]
            if pool:
                c = rng.choice(pool)
            else:  # no recorded reply of this kind: a turn that asks something
                asks = [
                    c
                    for c in cands["_asks"]
                    if (c["run_id"], c["point"]) not in used[tid]
                ]
                c = dict(
                    rng.choice(asks or cands["_asks"] or cands["_any"]),
                    match="heuristic",
                )
                last = run_turns[c["run_id"]][c["point"] - 1]["text"]
                yes_ops = [
                    a["operation"] for a in task["approvals"] if a["answer"] == "yes"
                ]
                all_ops = [a["operation"] for a in task["approvals"]]
                ops = yes_ops if kind.startswith("approve") else all_ops
                if ops:
                    c["op"] = max(ops, key=lambda o: (op_score(o, last), rng.random()))
            used[tid].add((c["run_id"], c["point"]))
            facts = {f["id"]: f for f in task["user_facts"]}
            fact = c.get("fact") or (rng.choice(sorted(facts)) if facts else None)
            if kind == "dont_know":
                meaning, fact_ids, op = FIXED["dont_know"], [], None
            elif kind == "reveal":
                meaning, fact_ids, op = facts[fact]["value"], [fact], None
            elif kind == "approve":
                meaning, fact_ids, op = approval(c["op"]), [], c["op"]
            elif kind == "refuse":
                meaning, fact_ids, op = FIXED["refuse"], [], c["op"]
            else:
                meaning = f"{approval(c['op'])}. {facts[fact]['value']}"
                fact_ids, op = [fact], c["op"]
            counts[kind] += 1
            corpus.append(
                {
                    "id": f"p{len(corpus):03d}",
                    "task_id": tid,
                    "kind": kind,
                    "meaning": meaning,
                    "fact_ids": fact_ids,
                    "operation": op,
                    "run_id": c["run_id"],
                    "point": c["point"],
                    "context_match": c["match"],
                    "real_reply": (
                        run_turns[c["run_id"]][c["point"]]["text"]
                        if c["point"] < len(run_turns[c["run_id"]])
                        else None
                    ),
                }
            )
    return corpus


def main():
    corpus = build()
    OUT.write_text(json.dumps(corpus, indent=1) + "\n")
    print(len(corpus), "pairs;", len({c["task_id"] for c in corpus}), "tasks")
    print(collections.Counter(c["kind"] for c in corpus))
    print(collections.Counter(c["context_match"] for c in corpus))


if __name__ == "__main__":
    main()
