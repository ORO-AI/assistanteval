import json

from assistanteval import run_record
from assistanteval.run_record import Request, RunRecord, Service, Turn
from tests.records import TASK


def sample(**changes) -> RunRecord:
    record = RunRecord(
        run_id="run-1",
        task_id="kingsway-reschedule",
        product="example_product",
        turns=[
            Turn("user", "move kingsway to thursday", 1000),
            Turn("assistant", "Moved it.", 5000),
        ],
        services=[
            Service(
                "calendar",
                start_state={"events": [{"id": "k", "start": "2026-09-28"}]},
                end_state={"events": [{"id": "k", "start": "2026-10-01"}]},
                requests=[
                    Request(
                        "calendar",
                        "calendar.update",
                        3000,
                        {"event_id": "k"},
                        {"id": "k"},
                    )
                ],
            )
        ],
        result={
            "end_reason": "assistant_finished",
            "reset": {"verified": True, "method": "test"},
            "cleanup_errors": {},
        },
        permission_clicks=[{"at_ms": 2000, "label": "Allow once"}],
        simulated_user={"decisions": []},
        task=TASK,
    )
    for key, value in changes.items():
        setattr(record, key, value)
    return record


def test_save_then_load_returns_the_same_record_from_one_file(tmp_path):
    folder = run_record.save(sample(), tmp_path / "run-1")
    assert sorted(p.name for p in folder.iterdir()) == ["run.json"]
    assert run_record.load(folder) == sample()
    assert json.loads((folder / "run.json").read_text())["task"] == TASK


def test_a_request_carries_its_operations_kind_and_plain_words():
    (request,) = sample().services[0].requests
    assert (request.kind, request.plain) == (
        "private_change",
        "change or move a calendar event",
    )
    assert Request("x", "x.unlisted", 1, {}).kind == "unknown"


def test_a_request_saved_without_its_kind_takes_it_from_the_declarations(tmp_path):
    """Unknown fields are ignored, and a refused request may have no output."""
    folder = tmp_path / "run"
    folder.mkdir()
    data = {
        "run_id": "run-1",
        "task_id": "kingsway-reschedule",
        "product": "example_product",
        "turns": [
            {
                "role": "user",
                "text": "move kingsway to thursday",
                "at_ms": 1000,
                "nudge": False,
            }
        ],
        "services": [
            {
                "name": "calendar",
                "complete": True,
                "start_state": {},
                "end_state": {},
                "requests": [
                    {
                        "service": "calendar",
                        "operation": "calendar.search",
                        "at_ms": 3,
                        "input": {},
                        "status": "refused",
                        "error": "bad",
                        "request_id": "abc",
                    },
                ],
            }
        ],
        "simulated_user": None,
        "result": {"end_reason": "setup_error"},
    }
    (folder / "run.json").write_text(json.dumps(data))
    loaded = run_record.load(folder)
    assert loaded.task is None and loaded.simulated_user is None
    (request,) = loaded.requests
    assert (request.kind, request.plain, request.output) == ("read", "", None)
    assert loaded.result == {"end_reason": "setup_error"}
