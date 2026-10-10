"""Summarise the live A/B runs under ~/ae-exp-data/live/<arm>/<model>/<task>-<rep>."""

import collections
import json
from pathlib import Path

LIVE = Path.home() / "ae-exp-data" / "live"
HELD, GAVE = (
    "simulated user read the assistant correctly",
    "simulated user gave a yes or a fact only when due",
)

for arm in sorted(p.name for p in LIVE.iterdir()):
    runs = sorted(LIVE.glob(f"{arm}/*/*"))
    stats = collections.Counter()
    outcomes = collections.Counter()
    per_task = collections.defaultdict(list)
    for d in runs:
        rec, ver = d / "run" / "run.json", d / "grade" / "verdict.json"
        if not rec.is_file() or not ver.is_file():
            stats["incomplete"] += 1
            continue
        r, v = json.loads(rec.read_text()), json.loads(ver.read_text())
        decisions = (r.get("simulated_user") or {}).get("decisions", [])
        actions = collections.Counter(x["action"] for x in decisions)
        replies = [
            x for x in decisions if x.get("text") and x["action"] not in ("nudge",)
        ]
        stats["runs"] += 1
        stats["user replies"] += len(replies)
        stats["dont_know"] += actions["dont_know"]
        stats["decline"] += actions["decline"]
        stats["reworded"] += sum(bool(x.get("rewrites", 0) < 2) for x in replies)
        checks = {c["name"]: c["result"] for c in v["validity"]}
        stats["reading disputed"] += (
            checks.get(HELD) != "pass" or checks.get(GAVE) != "pass"
        )
        stats["setup_error"] += r["result"]["end_reason"] == "setup_error"
        outcomes[v["outcome"]] += 1
        per_task[d.name.rsplit("-", 1)[0]].append(v["outcome"])
    print(f"== {arm}: {dict(stats)}")
    print(f"   outcomes: {dict(outcomes)}")
    for t, o in sorted(per_task.items()):
        print(f"   {t:28} {collections.Counter(o)}")
