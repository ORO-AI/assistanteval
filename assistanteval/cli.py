"""assistanteval run | grade | report | annotate | serve | harbor-export | harbor-verify: each command's own --help says more."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "run": "assistanteval.run_task",
    "grade": "assistanteval.grader.__main__",
    "report": "assistanteval.grader.report",
    "annotate": "assistanteval.grader.annotate",
    "serve": "assistanteval.remote",
    "harbor-export": "assistanteval.harbor.export",
    "harbor-verify": "assistanteval.harbor.verify",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: assistanteval {{{','.join(COMMANDS)}}} ...", file=sys.stderr)
        return 2
    return importlib.import_module(COMMANDS[argv[0]]).main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
