"""Print every outcome or gate that differs between two grade_offline.py outputs.

uv run python experiments/derived/compare.py <before.json> <after.json>
"""

import json
import sys
from collections import Counter
from pathlib import Path


def main(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    changed = Counter()
    for record, b in before.items():
        a = after[record]
        if a["outcome"] != b["outcome"]:
            print(f"OUTCOME {record}: {b['outcome']} -> {a['outcome']}")
            changed["outcome"] += 1
        old = Counter(map(json.dumps, b["gates"]))
        new = Counter(map(json.dumps, a["gates"]))
        for gate in old - new:
            print(f"  - {record}: {gate}")
            changed["gate_removed"] += 1
        for gate in new - old:
            print(f"  + {record}: {gate}")
            changed["gate_added"] += 1
    print(f"records: {len(before)}, changes: {dict(changed)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
