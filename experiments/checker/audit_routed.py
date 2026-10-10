"""Have independent models audit the decisions the new simulated user routed away from "i don't know".

uv run python experiments/checker/audit_routed.py ~/ae-exp-data/routed.json
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe import published  # noqa: E402

from assistanteval.endpoints import Endpoint  # noqa: E402
from assistanteval.grader.judges import CONTEXT_TURNS, LLMJudge, Question, read  # noqa: E402
from assistanteval.run_record import describe  # noqa: E402
from assistanteval.simulated_user import OFFERS_MORE, PROPOSES, written_since_reply  # noqa: E402

MODELS = ["anthropic/claude-sonnet-5.5", "google/gemini-3.1-pro-preview"]


def question(row):
    task, turns, _ = published(row["run"])
    seen = turns[: row["point"]]
    turn = {
        "assistant_turn": "\n\n".join(t.text for t in written_since_reply(seen)),
        "earlier_messages": [
            f"{t.role}: {t.text}" for t in seen[-CONTEXT_TURNS - 1 : -1]
        ],
    }
    if row["action"] == "decline":
        return Question("audit_offer", "yes_no", OFFERS_MORE, turn)
    action = describe(row["operation"])[1] or row["operation"]
    return Question("audit_proposal", "yes_no", PROPOSES, {**turn, "action": action})


def main() -> int:
    rows = json.loads(Path(sys.argv[1]).read_text())
    key = os.environ["OPENROUTER_API_KEY"]
    judges = {
        m: LLMJudge(Endpoint(m, "https://openrouter.ai/api/v1", key)) for m in MODELS
    }

    def audit(row):
        q = question(row)
        out = {}
        for m, j in judges.items():
            try:
                v, sure = read("yes_no", j(q))
                out[m] = "yes" if sure and v else "no" if sure else "unsure"
            except Exception as exc:  # noqa: BLE001
                out[m] = type(exc).__name__
        return {
            "run": row["run"],
            "action": row["action"],
            "operation": row.get("operation"),
            "audit": out,
        }

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(audit, rows))
    Path(sys.argv[1]).with_suffix(".audit.json").write_text(
        json.dumps(results, indent=1)
    )
    for kind in ("decline", "routed approval or refusal"):
        sub = [r for r in results if (r["action"] == "decline") == (kind == "decline")]
        both_yes = sum(all(v == "yes" for v in r["audit"].values()) for r in sub)
        any_no = sum(any(v == "no" for v in r["audit"].values()) for r in sub)
        print(
            f"{kind}: {len(sub)} | both models confirm: {both_yes} | at least one says no: {any_no}"
        )
        for m in MODELS:
            print(
                f"   {m:32} "
                + str(
                    {
                        v: sum(r["audit"][m] == v for r in sub)
                        for v in ("yes", "no", "unsure")
                    }
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
