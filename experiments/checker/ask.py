"""Ask independent models the grader re-check questions the official judges left unresolved.

uv run python experiments/checker/ask.py ~/ae-exp-data/unresolved.json --out ~/ae-exp-data/results/checker.json
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistanteval.endpoints import Endpoint  # noqa: E402
from assistanteval.grader.judges import LLMJudge, Question, read  # noqa: E402

MODELS = ["anthropic/claude-sonnet-5.5", "google/gemini-3.1-pro-preview"]


def main() -> int:
    qs = json.loads(Path(sys.argv[1]).read_text())
    key = os.environ["OPENROUTER_API_KEY"]
    judges = {
        m: LLMJudge(Endpoint(m, "https://openrouter.ai/api/v1", key)) for m in MODELS
    }

    def ask(q):
        question = Question(
            q["id"], q["kind"], q["text"], q["context"], q.get("criteria")
        )
        out = {}
        for m, judge in judges.items():
            try:
                p = judge(question)
                value, sure = read(q["kind"], p)
                out[m] = {"p": p, "value": value, "sure": sure}
            except Exception as exc:  # noqa: BLE001
                out[m] = {"error": type(exc).__name__}
        return {**q, "checkers": out}

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(ask, qs))
    Path(sys.argv[3]).write_text(json.dumps(rows, indent=1))
    for kind in ("user_rule", "user_read"):
        sub = [r for r in rows if r["id"].startswith(kind)]
        print(f"{kind}: {len(sub)} questions")
        for m in MODELS:
            ok = [r["checkers"][m] for r in sub if "error" not in r["checkers"][m]]
            sure = [c for c in ok if c["sure"]]
            agree = [
                1
                for r in sub
                if r["checkers"][m].get("sure")
                and r["checkers"][m]["value"] is r["confirms"]
            ]
            print(
                f"  {m:32} sure {len(sure)}/{len(sub)}, of which confirm the simulated user {len(agree)}"
            )
        both = [r for r in sub if all(r["checkers"][m].get("sure") for m in MODELS)]
        same = [
            r for r in both if len({r["checkers"][m]["value"] for m in MODELS}) == 1
        ]
        print(f"  both sure: {len(both)}, and agree with each other: {len(same)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
