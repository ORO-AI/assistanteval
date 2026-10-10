"""Ask the JUDGE the writer's same-meaning question for rewordings the reader was unsure of.

uv run python experiments/styles/judge_unsure.py   # reads raw_tight.jsonl, writes judge_unsure.json (local)
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistanteval.endpoints import required  # noqa: E402
from assistanteval.grader.judges import Question, judge_for, read  # noqa: E402

HERE = Path(__file__).parent


def main() -> int:
    rows = [json.loads(line) for line in (HERE / "raw_tight.jsonl").open()]
    judge = judge_for(required("JUDGE"))
    items = [
        (r["style"], r["meaning"], a["text"], a["reader_p_same"])
        for r in rows
        if r["style"] in ("none-tight", "terse-tight")
        for a in r["attempts"]
        if a.get("reader_p_same") is not None and 0.3 < a["reader_p_same"] < 0.7
    ]

    def ask(item):
        style, meaning, text, p = item
        q = Question(
            "user_checks_rewording",
            "yes_no",
            "Does `rewording` say the same thing as `message` and add nothing to it?",
            {"message": meaning, "rewording": text},
        )
        value, sure = read("yes_no", judge(q))
        return {
            "style": style,
            "message": meaning,
            "rewording": text,
            "reader_p": p,
            "judge": "yes" if sure and value else "no" if sure else "unsure",
        }

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(ask, items))
    (HERE / "judge_unsure.json").write_text(json.dumps(out, indent=1))
    for style in ("none-tight", "terse-tight"):
        sub = [o for o in out if o["style"] == style]
        print(
            style,
            len(sub),
            {k: sum(o["judge"] == k for o in sub) for k in ("yes", "no", "unsure")},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
