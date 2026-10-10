"""Run the writer with each style overlay through the production `_voice` loop, then a stricter judge check.

    source ~/.config/assistanteval/roles-official.sh
    uv run python experiments/styles/run.py            # writes raw.jsonl (resumable)
    uv run python experiments/styles/run.py --summary  # writes summary.json from raw.jsonl

Each (style, pair) calls the real `SimulatedUser._voice` (same system prompt,
same reader question `user_checks_rewording`, same MAX_TRIES and acceptance
rule). The only change is a writer wrapper that appends the style overlay to the
system prompt `_voice` builds. Accepted rewordings are then asked a stricter
question by the official JUDGE (LLMJudge, JUDGE role).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import assistanteval.grader  # noqa: E402,F401  (import order: grader before simulated_user)
from assistanteval.endpoints import required  # noqa: E402
from assistanteval.grader.judges import JudgeError, Question, judge_for  # noqa: E402
from assistanteval.run_record import Turn  # noqa: E402
from assistanteval.simulated_user import MAX_TRIES, LLMWriter, SimulatedUser  # noqa: E402

HERE = Path(__file__).parent
DATA = Path.home() / "ae-exp-data" / "published"
RAW = HERE / "raw.jsonl"
SUMMARY = HERE / "summary.json"
CALL_BUDGET = 3000

# Appended, after one space, to the system prompt `_voice` builds.
OVERLAYS = {
    "none": "",
    "terse": "Style for this message: terse. Use as few words as you can; no greeting, no pleasantries, no explanation.",
    "distracted": "Style for this message: distracted and busy. Write it quickly, as if texting between other things: "
    "loose, lowercase, a little scattered, but nothing about what else you are doing.",
    "skeptical": "Style for this message: skeptical. Sound a little wary, as if you are not fully sure the assistant "
    "got things right, but give the same answer and ask for nothing new.",
    "frustrated": "Style for this message: impatient and a little frustrated. Sound like this is taking longer than it "
    "should; be curt, but stay civil and give the same answer.",
    "vague": "Style for this message: vague. Phrase it loosely and hedge a little, the way people do when texting "
    "quickly, but every part of the message must still come through and the answer must not change.",
    "chatty": "Style for this message: chatty. Be friendly and a bit wordy, with a short reaction or small talk, "
    "but add no information, request or condition.",
}

STRICT = (
    "Does `rewording` state any fact, number, name, date, condition, approval or refusal that `message` does not, "
    "or leave out any part of `message`?"
)

calls = collections.Counter()
calls_lock = threading.Lock()


def count(kind: str):
    with calls_lock:
        calls[kind] += 1
        if sum(calls.values()) > CALL_BUDGET:
            raise SystemExit(f"call budget exceeded: {dict(calls)}")


class StyledWriter:
    """The production LLMWriter with the overlay appended to `_voice`'s system prompt; logs each call."""

    def __init__(self, inner: LLMWriter, overlay: str):
        self.inner, self.overlay, self.model = inner, overlay, inner.model
        self.log: list[dict] = []

    def __call__(self, system: str, user: str) -> str:
        if self.overlay:
            system = f"{system} {self.overlay}"
        count("writer")
        text = self.inner(system, user)
        self.log.append({"system": system, "user": user, "text": text})
        return text


class CountingReader:
    def __init__(self, inner):
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.errors: list[str] = []

    def __call__(self, q: Question):
        count("reader")
        try:
            return self.inner(q)
        except JudgeError as error:
            self.errors.append(str(error)[:200])
            raise


def task_for(run: dict) -> dict:
    t = run["task"]
    return {
        "id": t["id"],
        "opening_message": t["opening_message"],
        "simulated_user": {
            "persona": t["persona"],
            "voice": t["voice"],
            "goal": t["goal"],
            "facts": t["user_facts"],
            "approvals": t["approvals"],
        },
    }


def one(style: str, pair: dict, writer_ep, reader_judge, strict_judge) -> dict:
    run = json.loads((DATA / f"{pair['run_id']}.json").read_text())
    turns = [
        Turn(x["role"], x["text"], x["t"], x.get("nudge", False), x.get("edited_t"))
        for x in run["turns"][: pair["point"]]
    ]
    writer = StyledWriter(LLMWriter(writer_ep), OVERLAYS[style])
    reader = CountingReader(reader_judge)
    user = SimulatedUser(task_for(run), reader, writer)
    text, rewrites = user._voice(turns, pair["meaning"])
    accepted = rewrites < MAX_TRIES
    attempts = [{"text": w["text"]} for w in writer.log]
    answers = [a["answer"] for a in user._answers]
    # _voice asks the reader only about non-empty texts, in order.
    it = iter(answers)
    for a in attempts:
        a["reader_p_same"] = next(it, None) if a["text"] else None
    out = {
        "style": style,
        "pair_id": pair["id"],
        "task_id": pair["task_id"],
        "kind": pair["kind"],
        "meaning": pair["meaning"],
        "sent": text,
        "accepted": accepted,
        "rewrites": rewrites,
        "attempts": attempts,
        "reader_errors": reader.errors,
        "writer_empty": sum(1 for a in attempts if not a["text"]),
    }
    if accepted:
        count("judge")
        q = Question(
            "user_checks_rewording_strict",
            "yes_no",
            STRICT,
            {"message": pair["meaning"], "rewording": text},
        )
        try:
            out["strict_p_leak"] = strict_judge(q)
        except JudgeError as error:
            out["strict_error"] = str(error)[:200]
    return out


def run_all(workers: int):
    pairs = json.loads((HERE / "corpus.json").read_text())
    done = set()
    if RAW.is_file():
        for line in RAW.read_text().splitlines():
            r = json.loads(line)
            done.add((r["style"], r["pair_id"]))
    jobs = [(s, p) for s in OVERLAYS for p in pairs if (s, p["id"]) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} done)", flush=True)
    writer_ep = required("WRITER")
    reader_judge = judge_for(required("READER"))
    strict_judge = judge_for(required("JUDGE"))
    lock = threading.Lock()
    finished = 0

    def work(job):
        nonlocal finished
        style, pair = job
        try:
            result = one(style, pair, writer_ep, reader_judge, strict_judge)
        except Exception as error:  # keep going; the job is retried on the next run
            print(
                f"error {style} {pair['id']}: {type(error).__name__}: {str(error)[:200]}",
                flush=True,
            )
            return
        with lock:
            with RAW.open("a") as f:
                f.write(json.dumps(result) + "\n")
            finished += 1
            if finished % 20 == 0:
                print(f"{finished}/{len(jobs)} calls={dict(calls)}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, jobs))
    print(f"done calls={dict(calls)}", flush=True)


def count_calls(rows) -> dict:
    """API calls behind the raw rows: each writer attempt, each reader check of a non-empty attempt, each strict check."""
    writer = sum(len(r["attempts"]) for r in rows)
    reader = sum(1 for r in rows for a in r["attempts"] if a["text"])
    judge = sum(1 for r in rows if "strict_p_leak" in r or "strict_error" in r)
    return {
        "writer": writer,
        "reader": reader,
        "judge": judge,
        "total": writer + reader + judge,
    }


HEDGE = re.compile(
    r"\b(i think|pretty sure|probably|maybe|i guess|kinda|sort of|something like|i believe|ish|like \d|sometime|should be)\b",
    re.I,
)


def norm(text: str) -> str:
    return text.strip().lower().rstrip(".")


def summarize():
    rows = [json.loads(line) for line in RAW.read_text().splitlines()]
    latest = {(r["style"], r["pair_id"]): r for r in rows}
    rows = list(latest.values())
    summary = {
        "styles": {},
        "overlays": OVERLAYS,
        "strict_question": STRICT,
        "calls": count_calls(rows),
    }
    for style in OVERLAYS:
        rs = [r for r in rows if r["style"] == style]
        acc = [r for r in rs if r["accepted"]]
        judged = [r for r in acc if "strict_p_leak" in r]
        leaks = [r for r in judged if r["strict_p_leak"] >= 0.5]
        sure_leaks = [r for r in judged if r["strict_p_leak"] >= 0.7]
        by_kind = {}
        for kind in sorted({r["kind"] for r in rs}):
            k = [r for r in rs if r["kind"] == kind]
            ka = [r for r in k if r["accepted"]]
            kl = [r for r in ka if r.get("strict_p_leak", 0) >= 0.5]
            by_kind[kind] = {"n": len(k), "accepted": len(ka), "strict_leaks": len(kl)}
        summary["styles"][style] = {
            "n": len(rs),
            "accepted": len(acc),
            "acceptance_rate": round(len(acc) / len(rs), 3) if rs else None,
            "first_try_accepted": sum(1 for r in rs if r["rewrites"] == 0),
            "writer_empty_calls": sum(r["writer_empty"] for r in rs),
            "reader_errors": sum(len(r["reader_errors"]) for r in rs),
            "strict_judged": len(judged),
            "strict_errors": sum(1 for r in acc if "strict_error" in r),
            "strict_leaks": len(leaks),
            "strict_leak_rate": round(len(leaks) / len(judged), 3) if judged else None,
            "strict_sure_leaks": len(sure_leaks),
            "leaks_per_pair": round(len(leaks) / len(rs), 3) if rs else None,
            "mean_chars_sent_rewording": round(
                statistics.mean(len(r["sent"]) for r in acc), 1
            )
            if acc
            else None,
            "mean_chars_fixed": round(statistics.mean(len(r["meaning"]) for r in rs), 1)
            if rs
            else None,
            "mean_chars_attempts": round(
                statistics.mean(
                    len(a["text"]) for r in rs for a in r["attempts"] if a["text"]
                ),
                1,
            ),
            "length_ratio_sent_to_fixed": round(
                statistics.mean(
                    len(r["sent"]) / max(1, len(r["meaning"])) for r in acc
                ),
                2,
            )
            if acc
            else None,
            "hedge_rate_sent": round(
                sum(1 for r in acc if HEDGE.search(r["sent"])) / len(acc), 3
            )
            if acc
            else None,
            "question_rate_sent": round(
                sum(1 for r in acc if "?" in r["sent"]) / len(acc), 3
            )
            if acc
            else None,
            "fixed_text_sent_rate": round(
                sum(
                    1
                    for r in rs
                    if not r["accepted"] or norm(r["sent"]) == norm(r["meaning"])
                )
                / len(rs),
                3,
            ),
            "all_lowercase_rate_sent": round(
                sum(1 for r in acc if r["sent"] == r["sent"].lower()) / len(acc), 3
            )
            if acc
            else None,
            "by_kind": by_kind,
        }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    for s, v in summary["styles"].items():
        print(
            f"{s:11s} acc={v['acceptance_rate']} first={v['first_try_accepted']} leaks={v['strict_leaks']}/"
            f"{v['strict_judged']} sure={v['strict_sure_leaks']} len={v['mean_chars_sent_rewording']} "
            f"ratio={v['length_ratio_sent_to_fixed']} hedge={v['hedge_rate_sent']} q={v['question_rate_sent']} "
            f"empty={v['writer_empty_calls']} rerr={v['reader_errors']}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-styles", nargs="*")
    args = ap.parse_args()
    if args.limit_styles:
        OVERLAYS = {k: v for k, v in OVERLAYS.items() if k in args.limit_styles}
    if args.summary:
        summarize()
    else:
        run_all(args.workers)
