"""Tightened base writer prompt: none-tight, terse-tight and distracted-tight on the 120-pair corpus plus decline pairs.

    source ~/.config/assistanteval/roles-official.sh
    uv run python experiments/styles/tight.py decline    # writes corpus_decline.json (no API calls)
    uv run python experiments/styles/tight.py run        # writes raw_tight.jsonl (resumable)
    uv run python experiments/styles/tight.py summary    # writes summary_tight.json

Same `_voice` mechanics as run.py (real SimulatedUser._voice, Jev check, 2 tries).
The writer wrapper swaps `_voice`'s instruction sentences for TIGHT, keeps the
persona/voice framing, then appends the style overlay (if any).
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as base  # noqa: E402  (also sets up the assistanteval imports)
from corpus import runs as published_runs  # noqa: E402

from assistanteval.endpoints import required  # noqa: E402
from assistanteval.grader.judges import judge_for  # noqa: E402
from assistanteval.run_record import Turn  # noqa: E402
from assistanteval.simulated_user import MAX_TRIES, LLMWriter, SimulatedUser  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "raw_tight.jsonl"
DECLINE = "no thanks, that's all for now"
DECLINE_PAIRS = 20

# The instruction sentences `_voice` writes after the persona and voice.
CURRENT = (
    "Rewrite the given message in your voice. Keep exactly the same meaning. "
    "Do not add any fact, number, name, date, request, or approval."
)
TIGHT = (
    "Rewrite the given message in your voice. Keep exactly the same meaning and leave nothing out: "
    "if it says yes to an action, name that action; keep every name, number, address, email, time and date "
    "word for word. Do not add a greeting, sign-off, question, promise, instruction, condition, reason or "
    "opinion, and do not add any fact, number, name, date, request, or approval."
)
ARMS = {
    "none-tight": "",
    "terse-tight": base.OVERLAYS["terse"],
    "distracted-tight": base.OVERLAYS["distracted"],
}
base.CALL_BUDGET = 1600

OFFER = re.compile(
    r"(anything else|want me to|would you like me to|should i also|shall i also|happy to (also|help)|"
    r"let me know if you('d| would) like|do you want me to)",
    re.I,
)
DECLINED = re.compile(
    r"^\W*(no thanks|nope|no,? that'?s (all|it)|that'?s all|all good|no need|nah)", re.I
)


def build_decline():
    """One pair per task (20 tasks): an assistant turn that ends by offering more help."""
    rng = random.Random(20261011)
    by_task: dict[str, list] = {}
    for r in published_runs():
        turns = r["turns"]
        for i, t in enumerate(turns):
            if t["role"] == "assistant" and OFFER.search(t["text"][-300:]):
                nxt = turns[i + 1]["text"] if i + 1 < len(turns) else None
                by_task.setdefault(r["task"]["id"], []).append(
                    {
                        "run_id": r["id"],
                        "point": i + 1,
                        "declined": bool(nxt and DECLINED.search(nxt)),
                        "real_reply": nxt,
                    }
                )
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    pairs = []
    for tid in tasks[:DECLINE_PAIRS]:
        cands = by_task[tid]
        c = rng.choice([x for x in cands if x["declined"]] or cands)
        pairs.append(
            {
                "id": f"d{len(pairs):03d}",
                "task_id": tid,
                "kind": "decline",
                "meaning": DECLINE,
                "fact_ids": [],
                "operation": None,
                "run_id": c["run_id"],
                "point": c["point"],
                "context_match": "real_reply" if c["declined"] else "heuristic",
                "real_reply": c["real_reply"],
            }
        )
    (HERE / "corpus_decline.json").write_text(json.dumps(pairs, indent=1) + "\n")
    print(
        len(pairs),
        "decline pairs;",
        sum(p["context_match"] == "real_reply" for p in pairs),
        "at a real decline",
    )


class TightWriter(base.StyledWriter):
    def __call__(self, system: str, user: str) -> str:
        if CURRENT not in system:
            raise SystemExit("_voice's prompt changed: CURRENT no longer matches")
        return super().__call__(system.replace(CURRENT, TIGHT), user)


def one(arm: str, pair: dict, writer_ep, reader_judge) -> dict:
    r = json.loads((base.DATA / f"{pair['run_id']}.json").read_text())
    turns = [
        Turn(x["role"], x["text"], x["t"], x.get("nudge", False), x.get("edited_t"))
        for x in r["turns"][: pair["point"]]
    ]
    writer = TightWriter(LLMWriter(writer_ep), ARMS[arm])
    reader = base.CountingReader(reader_judge)
    user = SimulatedUser(base.task_for(r), reader, writer)
    text, rewrites = user._voice(turns, pair["meaning"])
    attempts = [{"text": w["text"]} for w in writer.log]
    answers = iter(a["answer"] for a in user._answers)
    for a in attempts:
        a["reader_p_same"] = next(answers, None) if a["text"] else None
    return {
        "style": arm,
        "pair_id": pair["id"],
        "task_id": pair["task_id"],
        "kind": pair["kind"],
        "meaning": pair["meaning"],
        "sent": text,
        "accepted": rewrites < MAX_TRIES,
        "rewrites": rewrites,
        "attempts": attempts,
        "reader_errors": reader.errors,
        "writer_empty": sum(1 for a in attempts if not a["text"]),
        "system_prompt": writer.log[0]["system"] if writer.log else None,
    }


def pairs() -> list[dict]:
    return json.loads((HERE / "corpus.json").read_text()) + json.loads(
        (HERE / "corpus_decline.json").read_text()
    )


def run_all(workers: int = 8):
    done = set()
    if RAW.is_file():
        done = {
            (r["style"], r["pair_id"])
            for r in map(json.loads, RAW.read_text().splitlines())
        }
    jobs = [(a, p) for a in ARMS for p in pairs() if (a, p["id"]) not in done]
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    writer_ep, reader_judge = required("WRITER"), judge_for(required("READER"))
    lock, finished = threading.Lock(), [0]

    def work(job):
        arm, pair = job
        try:
            result = one(arm, pair, writer_ep, reader_judge)
        except Exception as error:
            print(
                f"error {arm} {pair['id']}: {type(error).__name__}: {str(error)[:200]}",
                flush=True,
            )
            return
        with lock:
            with RAW.open("a") as f:
                f.write(json.dumps(result) + "\n")
            finished[0] += 1
            if finished[0] % 20 == 0:
                print(f"{finished[0]}/{len(jobs)} calls={dict(base.calls)}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, jobs))
    print(f"done calls={dict(base.calls)}", flush=True)


def stats(rows: list[dict]) -> dict:
    acc = [r for r in rows if r["accepted"]]
    return {
        "n": len(rows),
        "accepted": len(acc),
        "acceptance_rate": round(len(acc) / len(rows), 3) if rows else None,
        "first_try_accepted": sum(1 for r in rows if r["rewrites"] == 0),
        "fixed_text_sent_rate": round(
            sum(
                1
                for r in rows
                if not r["accepted"] or base.norm(r["sent"]) == base.norm(r["meaning"])
            )
            / len(rows),
            3,
        )
        if rows
        else None,
        "mean_chars_sent_rewording": round(
            statistics.mean(len(r["sent"]) for r in acc), 1
        )
        if acc
        else None,
        "mean_chars_attempts": round(
            statistics.mean(
                len(a["text"]) for r in rows for a in r["attempts"] if a["text"]
            ),
            1,
        )
        if rows
        else None,
        "all_lowercase_rate_sent": round(
            sum(1 for r in acc if r["sent"] == r["sent"].lower()) / len(acc), 3
        )
        if acc
        else None,
        "writer_empty_calls": sum(r["writer_empty"] for r in rows),
        "by_kind": {
            k: f"{sum(r['accepted'] for r in rows if r['kind'] == k)}/{sum(1 for r in rows if r['kind'] == k)}"
            for k in sorted({r["kind"] for r in rows})
        },
    }


def summarize():
    tight = list(
        {
            (r["style"], r["pair_id"]): r
            for r in map(json.loads, RAW.read_text().splitlines())
        }.values()
    )
    old = list(
        {
            (r["style"], r["pair_id"]): r
            for r in map(json.loads, base.RAW.read_text().splitlines())
        }.values()
    )
    out = {
        "current_instructions": CURRENT,
        "tight_instructions": TIGHT,
        "overlays": ARMS,
        "calls": base.count_calls(tight),
    }
    out["arms"] = {}
    for arm in ARMS:
        rows = [r for r in tight if r["style"] == arm]
        out["arms"][arm] = {
            "all": stats(rows),
            "corpus120": stats([r for r in rows if r["kind"] != "decline"]),
            "decline": stats([r for r in rows if r["kind"] == "decline"]),
        }
    out["earlier"] = {
        s: stats([r for r in old if r["style"] == s])
        for s in ("none", "terse", "distracted")
    }
    (HERE / "summary_tight.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, v in [(k, a["corpus120"]) for k, a in out["arms"].items()] + list(
        out["earlier"].items()
    ):
        print(
            f"{name:17s} acc={v['accepted']}/{v['n']} first={v['first_try_accepted']} fixed={v['fixed_text_sent_rate']} "
            f"len={v['mean_chars_sent_rewording']} att={v['mean_chars_attempts']} lc={v['all_lowercase_rate_sent']} "
            f"empty={v['writer_empty_calls']} {v['by_kind']}"
        )
    for k, a in out["arms"].items():
        print(
            f"{k:17s} decline acc={a['decline']['accepted']}/{a['decline']['n']} len={a['decline']['mean_chars_sent_rewording']}"
        )
    print(out["calls"])


if __name__ == "__main__":
    {"decline": build_decline, "run": run_all, "summary": summarize}[sys.argv[1]]()
