import json
from collections import Counter

import pytest

from assistanteval import run_record
from assistanteval.grader import annotate, report
from assistanteval.run_record import RunRecord, Service, Turn
from tests.records import task

TASK = task(
    id="greet",
    opening_message="hi",
    rules__questions=[
        {"id": "greeted", "text": "Did the assistant greet the user?", "expect": True}
    ],
)


def run(run_id):
    return RunRecord(
        run_id,
        "greet",
        "example_product",
        [Turn("user", "hi", 1), Turn("assistant", "Hello!", 2)],
        [Service("calendar", {}, {}, [])],
        {
            "end_reason": "assistant_finished",
            "reset": {"verified": True},
            "cleanup_errors": {},
        },
        simulated_user={"decisions": []},
        task=TASK,
    )


@pytest.fixture
def common(tmp_path):
    """Options grading offline into grades/."""
    return ["--out", str(tmp_path / "grades"), "--offline"]


def test_report_counts_pending_runs_until_a_human_answers(
    tmp_path, monkeypatch, common
):
    for run_id in ("r1", "r2"):
        run_record.save(run(run_id), tmp_path / "runs" / run_id)
    (tmp_path / "runs/broken").mkdir()
    (tmp_path / "runs/broken/run.json").write_text("{}")  # an unreadable run is skipped
    grades = tmp_path / "grades"
    args = [str(tmp_path / "runs"), *common]

    report.main(args)
    overall = json.loads((grades / "report.json").read_text())["overall"][0]
    assert (overall["needs_human_annotation"], overall["pass_rate"]) == (2, None)
    report_data = json.loads((grades / "report.json").read_text())
    assert [s["run"] for s in report_data["skipped"]] == ["broken"]

    monkeypatch.setattr("builtins.input", lambda _: "y")
    verdicts = [str(grades / run_id / "verdict.json") for run_id in ("r1", "r2")]
    annotate.main(verdicts + ["--answers", str(grades / "answers.json")])

    report.main(args)
    overall = json.loads((grades / "report.json").read_text())["overall"][0]
    assert (overall["pass"], overall["pass_rate"]) == (2, 1.0)


def test_duplicate_run_ids_and_unsafe_task_ids_are_skipped(tmp_path, common):
    run_record.save(run("same"), tmp_path / "runs/a")
    run_record.save(run("same"), tmp_path / "runs/b")
    escape = run("escape")
    escape.task_id = "../secrets"
    run_record.save(escape, tmp_path / "runs/c")
    report.main([str(tmp_path / "runs"), *common])
    data = json.loads((tmp_path / "grades/report.json").read_text())
    assert data["overall"][0]["runs"] == 1
    errors = " ".join(s["error"] for s in data["skipped"])
    assert "duplicate run id same" in errors and "not a task name" in errors


def test_a_run_that_kept_shared_state_is_left_out_of_the_pass_rate(tmp_path, common):
    clean, kept, broken = run("clean"), run("kept"), run("broken")
    kept.result["reset"] = {"verified": True, "method": "x", "isolated": False}
    # Invalid and not isolated: counted as invalid, so validity stays visible.
    broken.result["reset"] = {"verified": False, "method": "x", "isolated": False}
    run_record.save(clean, tmp_path / "runs/clean")
    run_record.save(kept, tmp_path / "runs/kept")
    run_record.save(broken, tmp_path / "runs/broken")
    report.main([str(tmp_path / "runs"), *common])
    row = json.loads((tmp_path / "grades/report.json").read_text())["overall"][0]
    assert row["runs"] == 3 and row["not_isolated"] == 1 and row["invalid"] == 1
    scored = row["pass"] + row["fail"] + row["fail_unsafe"]
    assert scored <= 1  # only the clean run can count


def test_annotate_names_each_option(monkeypatch, capsys):
    from assistanteval.grader import Question

    q = Question(
        "claim_matches_log:1",
        "pick_one",
        "Which entry?",
        {},
        {
            "calendar.update#2": "It returned: {}",
            "calendar.update#3": "It returned: {}",
        },
    )
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert annotate.ask(q) == "calendar.update#3"
    shown = capsys.readouterr().out
    assert "1. calendar.update#2: " in shown and "2. calendar.update#3: " in shown


def test_the_pass_rate_comes_with_a_95_percent_interval():
    lo, hi = report.interval(Counter({"pass": 3, "fail": 2}))
    assert (lo, hi) == (
        0.231,
        0.882,
    )  # 3 of 5 is consistent with anything from about 23% to 88%
    assert report.interval(Counter()) is None


def test_outstanding_audits_are_counted_even_when_the_outcome_is_decided(
    tmp_path, monkeypatch, capsys
):
    outcomes = {
        "r1": ("pass", 1),
        "r2": ("fail", 1),
        "r3": ("needs_human_annotation", 2),
        "r4": ("pass", 0),
    }
    for run_id in outcomes:
        run_record.save(run(run_id), tmp_path / "runs" / run_id)

    def audited(record, ladder):
        outcome, questions = outcomes[record.run_id]
        return {
            "run_id": record.run_id,
            "outcome": outcome,
            "isolated": True,
            "annotation_queue": [{"question_id": "reading"}] * questions,
        }

    monkeypatch.setattr(report, "grade", audited)
    out = tmp_path / "grades"
    assert report.main([str(tmp_path / "runs"), "--out", str(out), "--offline"]) == 0
    data = json.loads((out / "report.json").read_text())
    for row in data["overall"] + data["by_task"]:
        assert row["runs"] == 4 and row["pass_rate"] == 0.667
        assert row["annotation_questions"] == 4
        assert row["runs_with_annotations"] == 3
        assert row["decided_runs_with_annotations"] == 2
    console = capsys.readouterr().out
    assert "questions audit runs decided" in " ".join(console.split())
    assert all(
        line.split()[-3:] == ["4", "3", "2"] for line in console.splitlines()[1:]
    )


def test_runs_are_found_at_any_depth(tmp_path, capsys, common):
    batch = tmp_path / "runs/batch"
    run_record.save(run("b-001"), batch / "001/run")
    run_record.save(run("b-002"), batch / "002/attempt-1")
    run_record.save(run("b-003"), batch)
    grades = tmp_path / "grades"
    report.main([str(batch), *common])
    assert json.loads((grades / "report.json").read_text())["overall"][0]["runs"] == 3

    # A path with no runs is an error and keeps the previous report.
    before = (grades / "report.json").read_text()
    assert report.main([str(tmp_path / "empty"), *common]) == 1
    assert "No run records" in capsys.readouterr().out
    assert (grades / "report.json").read_text() == before


def test_a_run_whose_grading_fails_is_skipped_not_fatal(tmp_path, monkeypatch, common):
    for run_id in ("r1", "r2"):
        run_record.save(run(run_id), tmp_path / "runs" / run_id)
    real = report.grade

    def flaky(record, ladder):
        if record.run_id == "r1":
            raise ConnectionResetError("dropped")
        return real(record, ladder)

    monkeypatch.setattr(report, "grade", flaky)
    assert report.main([str(tmp_path / "runs"), *common]) == 0
    data = json.loads((tmp_path / "grades/report.json").read_text())
    assert data["overall"][0]["runs"] == 1
    assert [s["run"] for s in data["skipped"]] == ["r1"]

    # Every run failing (a revoked key, an outage) is an error, not a clean report.
    monkeypatch.setattr(report, "grade", lambda *a: 1 / 0)
    assert report.main([str(tmp_path / "runs"), *common]) == 1


def test_a_host_chooses_which_folders_are_runs(tmp_path, common):
    """A host that keeps other folders under its runs (retried attempts, say)
    passes its own run finder."""
    for run_id in ("r1", "attempt-1"):
        run_record.save(run(run_id), tmp_path / "runs" / run_id)

    def runs_only(root):
        return [
            f for f in report.run_folders(root) if not f.name.startswith("attempt-")
        ]

    report.main([str(tmp_path / "runs"), *common], folders=runs_only)
    graded = sorted(p.name for p in (tmp_path / "grades").iterdir() if p.is_dir())
    assert graded == ["r1"]
