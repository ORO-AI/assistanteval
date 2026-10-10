"""Grade every live run record offline from saved official-judge answers.

    uv run python experiments/derived/grade_offline.py <records.txt> <answers.json> <out.json>

records.txt lists run.json paths, one per line. The answers file is only read
(offline grading adds nothing). Writes {record: {outcome, gates}} to out.json
so two grader versions can be compared with compare.py.
"""

import json
import sys
from pathlib import Path

from assistanteval import run_record
from assistanteval.grader import AnswerStore, Ladder, grade


def main(records: str, answers: str, out: str) -> None:
    store = AnswerStore(Path(answers))
    summary = {}
    for line in Path(records).read_text().split():
        verdict = grade(run_record.load(Path(line).parent), Ladder(store))
        summary[line] = {
            "grader_version": verdict["grader_version"],
            "outcome": verdict["outcome"],
            "gates": [
                [g["name"], g["about"], g["result"], g["decided_by"]]
                for g in verdict["gates"]
            ],
        }
    Path(out).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
