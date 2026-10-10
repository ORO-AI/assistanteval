"""The task runner, offline: a fake product, a fake user and fake services on a fake clock."""

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from assistanteval import dates, run_record, run_task
from assistanteval.contracts import EpisodeStepError, Message, ResetEvidence
from tests.conftest import catalog, render_task

TASK = json.loads((run_task.ROOT / "tasks/kingsway-reschedule-crew.json").read_text())
# A task with one dated fixture, for what the runner renders.
DATED = {
    "id": "demo-task",
    "environment": {
        "connectors": [
            {"service": "calendar", "fixture": "tests/fixtures/dated-calendar.json"}
        ],
        "timezone": "America/Vancouver",
        "dates": {"job": "today+1wd"},
    },
    "opening_message": "move kingsway from {{job:weekday}}",
}


class Clock:
    def __init__(self):
        self.ms = 1_000_000

    def now_ms(self):
        return self.ms

    def sleep(self, seconds):
        self.ms += int(seconds * 1000)


class FakeBot:
    """Shows scripted messages: `script` maps a user-turn count to the
    (id, text) list the product shows after that many user messages."""

    def __init__(self, clock, script=None, clicks=0, fail=None):
        self.clock, self.script, self.clicks, self.fail = (
            clock,
            script or {},
            clicks,
            fail,
        )
        self.sent = []

    def reset(self):
        if self.fail:
            raise self.fail
        return ResetEvidence(True, "fake_reset")

    def connect(self, connectors):
        return ["Calendar is connected."]

    def send(self, text):
        self.sent.append(text)

    def read(self):
        shown = self.script.get(len(self.sent), [])
        if callable(shown):
            shown = shown(self.clock)
        clicked = [{"label": "Allow once", "detail": "Calendar wants to update"}]
        clicked, self.clicks = clicked * self.clicks, 0
        return [Message(i, t) for i, t in shown], clicked


def assistant(bot, fail=None):
    @contextmanager
    def open_assistant(account, evidence_dir):
        if isinstance(fail, Exception):  # while opening
            raise fail
        try:
            yield bot
        finally:
            if fail == "close":
                raise EpisodeStepError("connector_cleanup_unverified")

    open_assistant.connector_kind = "openapi_api_key"
    return open_assistant


class FakeUser:
    """Replies with `moves` in order: "wait", "end", "end:finished", or text to send."""

    def __init__(self, moves=()):
        self.moves, self.seen = list(moves), []

    def opening(self):
        return "move kingsway to thursday"

    def reply(self, turns, at_ms):
        self.seen.append([t.text for t in turns])
        move = self.moves.pop(0) if self.moves else "end:finished"
        if move.startswith("end"):
            saw = "finished" if move == "end:finished" else "asks_other"
            return SimpleNamespace(action="end", text=None, decision={"saw": saw})
        if move == "wait":
            return SimpleNamespace(
                action="wait", text=None, decision={"saw": "no_progress"}
            )
        return SimpleNamespace(
            action="reveal", text=move, decision={"saw": "asks_for:x"}
        )

    def nudge(self, after_turn, at_ms):
        return SimpleNamespace(action="nudge", text="Any update?", decision={})

    def record(self):
        return {"decisions": len(self.seen)}


class FakeService:
    def __init__(self, fail_release=False):
        self.released, self.fail_release = False, fail_release

    def prepare(self):
        return SimpleNamespace(connector=SimpleNamespace(kind="openapi_api_key"))

    def state(self):
        return {"events": 1}

    def request_log(self):
        request = run_record.Request(
            "calendar", "calendar.update", 5, {"id": 1}, {"ok": True}
        )
        return [request], True

    def release(self):
        self.released = True
        if self.fail_release:
            raise EpisodeStepError("calendar_revocation_unverified")


def run(tmp_path, bot, user, clock, service=None, fail=None, task=TASK, **kw):
    service = service or FakeService()
    kw.setdefault("services_from", lambda *a: service)
    record = run_task.run_task(
        task,
        "fake",
        assistant(bot, fail),
        tmp_path / "run",
        user=user,
        now_ms=clock.now_ms,
        sleep=clock.sleep,
        **kw,
    )
    return record, service


def steady(*messages):
    """What the product shows, identical on every read."""
    return list(messages)


def test_a_finished_conversation_is_recorded_and_services_released(tmp_path):
    clock = Clock()
    bot = FakeBot(clock, {1: steady(("m1", "Done: Kingsway is Thursday."))}, clicks=2)
    record, service = run(tmp_path, bot, FakeUser(), clock)

    assert record.result["end_reason"] == "assistant_finished"
    assert record.result["reset"] == {
        "verified": True,
        "method": "fake_reset",
        "isolated": True,
    }
    assert [(t.role, t.text) for t in record.turns] == [
        ("user", "move kingsway to thursday"),
        ("assistant", "Done: Kingsway is Thursday."),
    ]
    assert len(record.permission_clicks) == 2
    # Each click keeps its time, its button and what the prompt asked.
    assert record.permission_clicks[0]["label"] == "Allow once"
    assert record.permission_clicks[0]["detail"] == "Calendar wants to update"
    assert "at_ms" in record.permission_clicks[0]
    assert [s["text"] for s in record.setup] == [
        "connector setup: calendar",
        "Calendar is connected.",
    ]
    assert record.services[0].requests[0].operation == "calendar.update"
    assert record.services[0].start_state == {"events": 1}
    assert service.released
    loaded = run_record.load(tmp_path / "run")
    assert (tmp_path / "run" / "run.json").is_file()
    assert loaded.result["end_reason"] == "assistant_finished"
    # The record holds the task it ran, and names the simulated user; the
    # result says when the run started and ended.
    # The task is the one rendered at the run's anchor, which the record keeps.
    zone = TASK["environment"]["timezone"]
    anchor = dates.parse_anchor(loaded.manifest["world"]["anchor"], zone)
    assert loaded.task == render_task(TASK, anchor)
    assert "{{" not in json.dumps(loaded.task) and "simulated_user" in loaded.manifest
    # Every caller's record names the runner and the model endpoints.
    assert loaded.manifest["runner_version"].startswith("assistanteval.run_task at ")
    assert "endpoints" in loaded.manifest
    assert loaded.result["started_at_ms"] < loaded.result["ended_at_ms"]


def test_a_message_counts_only_once_two_reads_agree(tmp_path):
    clock = Clock()
    growing = iter(["One mo", "One moment", "One moment", "One moment"])
    bot = FakeBot(clock, {1: lambda c: [("m1", next(growing, "One moment"))]})
    user = FakeUser()
    run(tmp_path, bot, user, clock)
    assert user.seen[0][-1] == "One moment"  # never the half-written "One mo"


def test_the_user_replies_once_a_burst_of_messages_is_over(tmp_path):
    clock = Clock()
    start = clock.ms

    def shown(c):  # the second message follows three seconds later
        burst = [("m1", "Done, I logged both:")]
        return burst + [("m2", "Thu: Luis, Kingsway")] * (c.ms - start >= 3000)

    user = FakeUser()
    run(tmp_path, FakeBot(clock, {1: shown}), user, clock)
    assert user.seen == [
        ["move kingsway to thursday", "Done, I logged both:", "Thu: Luis, Kingsway"]
    ]


def test_the_user_waits_while_the_assistant_is_still_answering(tmp_path):
    # A model writes "Let me find ... Ben's details", then calls tools for
    # longer than the settle time; that first message is not its answer.
    class Answering(FakeBot):
        answering, reads = True, 0

        def read(self):
            self.reads += 1
            done = self.reads >= 200  # the model answers past the nudge time
            self.answering = not done  # set with the messages it describes
            shown = [
                Message("m1", "We're stuck with 07:00. Let me find Ben's details.")
            ]
            if done:
                shown.append(Message("m2", "I emailed Ben about Thursday."))
            return shown, []

    user = FakeUser()
    run(tmp_path, Answering(Clock()), user, Clock())
    assert user.seen == [
        [
            "move kingsway to thursday",
            "We're stuck with 07:00. Let me find Ben's details.",
            "I emailed Ben about Thursday.",
        ]
    ]


def test_a_products_own_working_flag_does_not_hold_the_user(tmp_path):
    clock = Clock()
    bot = FakeBot(clock, {1: steady(("m1", "Which Thursday, the 1st or the 8th?"))})
    bot.working = (
        True  # a bare working flag is not read: adapters opt in with `answering`
    )
    user = FakeUser()
    record, _ = run(tmp_path, bot, user, clock)
    assert user.seen == [
        ["move kingsway to thursday", "Which Thursday, the 1st or the 8th?"]
    ]
    waited = clock.ms - record.turns[-1].at_ms  # from the question to the reply
    assert waited < 2 * run_task.SETTLE_SECONDS * 1000


def test_wait_keeps_reading_and_a_rewritten_message_is_updated_in_place(tmp_path):
    clock = Clock()
    start = clock.ms

    def shown(c):
        # Rewritten after the user has seen "Checking…" and chosen to wait.
        text = "Checking…" if c.ms - start < 10_000 else "Kingsway moved to Thursday."
        return [("m1", text)]

    bot = FakeBot(clock, {1: shown})
    record, _ = run(tmp_path, bot, FakeUser(["wait"]), clock)
    (turn,) = [t for t in record.turns if t.role == "assistant"]
    assert turn.text == "Kingsway moved to Thursday."
    # It keeps its first time and records when it changed.
    assert turn.at_ms < start + 10_000 <= turn.edited_at_ms
    assert record.result["end_reason"] == "assistant_finished"


def test_silence_is_nudged_and_three_unanswered_nudges_time_out(tmp_path):
    clock = Clock()
    record, _ = run(tmp_path, FakeBot(clock), FakeUser(), clock)
    assert record.result["end_reason"] == "timeout"
    assert [t.text for t in record.turns if t.nudge] == ["Any update?"] * 3


def test_the_user_can_send_the_same_text_twice(tmp_path):
    clock = Clock()
    asked = [("m1", "Which day?"), ("m2", "Which day again?")]
    bot = FakeBot(clock, {1: asked[:1], 2: asked, 3: [*asked, ("m3", "Done.")]})
    record, _ = run(tmp_path, bot, FakeUser(["thursday", "thursday"]), clock)
    assert bot.sent == ["move kingsway to thursday", "thursday", "thursday"]
    assert record.result["end_reason"] == "assistant_finished"


def test_user_turns_stop_at_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_task, "MAX_USER_TURNS", 2)
    clock = Clock()
    script = {n: [(f"m{k}", f"q{k}") for k in range(1, n + 1)] for n in range(1, 5)}
    record, _ = run(tmp_path, FakeBot(clock, script), FakeUser(["a", "b", "c"]), clock)
    assert record.result["end_reason"] == "turn_limit"
    assert sum(t.role == "user" for t in record.turns) == 2


@pytest.mark.parametrize(
    "failure, phase, account_ok",
    [
        (EpisodeStepError("open_refused"), "sign_in", False),
        (EpisodeStepError("reset_failed"), "reset", False),
        # An assistant that cannot be reached, or a network error, is
        # unverified: neither a refusal nor a success.
        (ConnectionError("connection refused"), "sign_in", None),
        (ConnectionResetError("connection reset"), "reset", None),
    ],
)
def test_an_open_or_reset_failure_counts_against_the_assistant_only_if_it_refused(
    tmp_path, failure, phase, account_ok
):
    clock = Clock()
    at = {phase: failure}
    bot = FakeBot(clock, fail=at.get("reset"))
    record, service = run(tmp_path, bot, FakeUser(), clock, fail=at.get("sign_in"))
    assert record.result["end_reason"] == "setup_error"
    assert record.result["phase"] == phase
    assert record.result["account_ok"] is account_ok
    assert record.result["opening_sent"] is False
    assert service.released


def test_a_failure_mid_conversation_is_a_setup_error_not_safe_to_retry(tmp_path):
    clock = Clock()

    def crash(c):
        raise EpisodeStepError("chat_closed")

    record, service = run(tmp_path, FakeBot(clock, {1: crash}), FakeUser(), clock)
    assert record.result["end_reason"] == "setup_error"
    assert record.result["error"] == "chat_closed"
    assert record.result["opening_sent"] is True
    assert record.result["account_ok"] is True
    assert service.released


def test_a_cleanup_failure_after_a_failed_run_keeps_the_runs_own_error(tmp_path):
    clock = Clock()

    def crash(c):
        raise EpisodeStepError("chat_closed")

    record, _ = run(
        tmp_path, FakeBot(clock, {1: crash}), FakeUser(), clock, fail="close"
    )
    assert (record.result["end_reason"], record.result["phase"]) == (
        "setup_error",
        "converse",
    )
    assert record.result["error"] == "chat_closed"
    assert record.result["cleanup_errors"] == {
        "assistant": "connector_cleanup_unverified"
    }


def test_a_stop_saves_the_record_and_releases(tmp_path):
    clock = Clock()

    def stop(c):
        raise run_task.Stop()

    record, service = run(tmp_path, FakeBot(clock, {1: stop}), FakeUser(), clock)
    assert record.result["error"] == "stopped"
    assert service.released
    assert (tmp_path / "run" / "run.json").exists()


def test_a_release_failure_is_recorded_not_raised(tmp_path):
    clock = Clock()
    bot = FakeBot(clock, {1: steady(("m1", "done"))})
    record, _ = run(
        tmp_path, bot, FakeUser(), clock, service=FakeService(fail_release=True)
    )
    assert record.result["cleanup_errors"] == {
        "calendar": "calendar_revocation_unverified"
    }


def test_a_failed_assistant_cleanup_keeps_the_finished_run(tmp_path):
    clock = Clock()
    bot = FakeBot(clock, {1: steady(("m1", "done"))})
    record, service = run(tmp_path, bot, FakeUser(), clock, fail="close")
    assert record.result["end_reason"] == "assistant_finished"
    assert "error" not in record.result
    assert record.result["cleanup_errors"] == {
        "assistant": "connector_cleanup_unverified"
    }
    assert service.released


FRIDAY_9AM = int(datetime(2026, 9, 25, 16, 0, 30, tzinfo=UTC).timestamp() * 1000)


def test_the_world_is_rendered_at_the_run_start_and_its_anchor_recorded(
    tmp_path, monkeypatch
):
    clock = Clock()
    clock.ms = FRIDAY_9AM
    seen = {}

    def services_from(name, fixture, kind):
        seen["path"], seen["fixture"] = fixture, fixture.read_bytes()
        return FakeService()

    def user_for(task):
        seen["task"] = task
        return FakeUser()

    monkeypatch.setattr(run_task, "simulated_user", user_for)
    bot = FakeBot(clock, {1: steady(("m1", "done"))})
    run(tmp_path, bot, None, clock, task=DATED, services_from=services_from)
    # The service gets the fixture rendered at the anchor, never the template.
    event = json.loads(seen["fixture"])["events"][0]
    assert event["id"] == "kingsway-2026-09-28"
    assert event["start"] == "2026-09-28T08:00:00-07:00"
    # So does the simulated user, who speaks for the rendered task.
    assert seen["task"]["opening_message"] == "move kingsway from monday"
    manifest = run_record.load(tmp_path / "run").manifest
    assert manifest["world"] == {
        "anchor": "2026-09-25T09:00:00-07:00",
        "fixtures": {"calendar": run_record.sha256(seen["fixture"])},
        # The rendered fixture itself, so the grader can check each service
        # started from it.
        "fixture_data": {"calendar": json.loads(seen["fixture"])},
    }
    assert not seen["path"].exists()  # the rendered copy is not left behind


def test_a_task_that_cannot_render_is_a_setup_error(tmp_path):
    clock = Clock()
    broken = {**DATED, "opening_message": "move it to {{nope}}"}
    record, _ = run(tmp_path, FakeBot(clock), FakeUser(), clock, task=broken)
    assert (record.result["end_reason"], record.result["phase"]) == (
        "setup_error",
        "world",
    )
    assert "world" not in run_record.load(tmp_path / "run").manifest


def test_a_simulated_user_that_cannot_start_is_a_setup_error(tmp_path, monkeypatch):
    def no_keys(task):
        raise KeyError("OPENROUTER_API_KEY")

    monkeypatch.setattr(run_task, "simulated_user", no_keys)
    clock = Clock()
    record, _ = run(tmp_path, FakeBot(clock), None, clock)
    assert record.result["end_reason"] == "setup_error"
    assert record.result["error"] == "KeyError"
    assert record.simulated_user is None


def test_a_replaced_half_written_message_does_not_block_nudges(tmp_path):
    """A product whose ids follow the text shows each partial text under a new
    id; once the text settles, the old partial ids must not count as writing."""
    clock = Clock()
    start = clock.ms

    def shown(c):
        if c.ms - start < 3000:
            text = "Kingsway mo" if (c.ms // 1000) % 2 else "Kingsway moved"
            return [(text, text)]
        return [("done", "Kingsway moved.")]

    record, _ = run(tmp_path, FakeBot(clock, {1: shown}), FakeUser(["wait"]), clock)
    assert [t.text for t in record.turns if t.role == "assistant"] == [
        "Kingsway moved."
    ]
    assert record.result["end_reason"] == "timeout"  # the nudges went out
    assert sum(t.nudge for t in record.turns) == 3


# The command line.


@pytest.fixture
def main(tmp_path, monkeypatch):
    """`run_task.main` with a fake assistant, fake services and a fake user
    on a fast clock. The assistant is named as module:factory."""
    monkeypatch.setattr(run_task, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(run_task, "SETTLE_SECONDS", 0)
    monkeypatch.setattr(run_task, "simulated_user", lambda task: FakeUser())
    task = catalog(tmp_path) / "task.json"
    task.write_text(json.dumps(TASK))
    runs = []

    def main(bot):
        factory = assistant(bot)
        factory.manifest = lambda: {"model_baseline": {"model": "openai/gpt-test"}}
        monkeypatch.setattr(run_task, "_resolve", lambda entry: factory)
        monkeypatch.setitem(
            run_task.run_task.__kwdefaults__, "services_from", lambda *a: FakeService()
        )
        out = tmp_path / f"run-{len(runs)}"
        runs.append(out)
        argv = [
            str(task),
            "--assistant",
            "my_harness.adapter:open_chat",
            "--out",
            str(out),
        ]
        return run_task.main(argv)

    main.runs = runs
    return main


def done_bot():
    return FakeBot(Clock(), {1: steady(("m1", "Done: Kingsway is Thursday."))})


def test_main_runs_an_assistant_from_any_module_and_records_its_endpoints(
    main, monkeypatch
):
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_API_KEY", "judge-secret")
    for role in ("READER", "WRITER", "BASELINE"):
        monkeypatch.delenv(f"ASSISTANTEVAL_{role}_MODEL", raising=False)
    assert main(done_bot()) == 0
    raw = (main.runs[0] / "run.json").read_text()
    manifest = json.loads(raw)["manifest"]
    assert manifest["model_baseline"] == {"model": "openai/gpt-test"}
    assert manifest["endpoints"] == {
        "judge": {
            "model": "judge-model",
            "api": "openai",
            "base_url": "http://localhost:8000/v1",
        }
    }
    assert "judge-secret" not in raw
    assert "account" not in manifest


def test_without_git_the_runner_version_is_the_recorded_revision(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(run_task.subprocess, "run", no_git)
    monkeypatch.setenv("ASSISTANTEVAL_REVISION", "abc123")
    assert run_task._runner_version() == "assistanteval.run_task at abc123"


def test_the_run_deadline_outlasts_the_conversation_deadline():
    # Time to save the record and release the services after the alarm.
    assert run_task.RUN_DEADLINE_SECONDS > run_task.DEADLINE_SECONDS


def test_a_hung_run_is_stopped_at_its_deadline_and_saves_its_record(main, monkeypatch):
    monkeypatch.setattr(run_task, "RUN_DEADLINE_SECONDS", 0.3)

    class Hung(FakeBot):
        def reset(self):
            time.sleep(5)  # interrupted by the deadline long before this ends
            pytest.fail("the run outlived its deadline")

    started = time.monotonic()
    assert main(Hung(Clock())) == 1
    assert time.monotonic() - started < 3
    result = json.loads((main.runs[0] / "run.json").read_text())["result"]
    assert result["error"] == "run_deadline" and result["phase"] == "reset"
    import signal

    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)  # alarm cleared


def test_an_answer_that_grows_in_an_answered_bubble_is_shown_to_the_user(tmp_path):
    clock = Clock()
    start = clock.ms

    def grows(c):
        # A product pauses mid-answer; the user replies "wait"-worthy text, then
        # the same bubble grows into the answer.
        text = (
            "Checking"
            if c.ms - start < 30_000
            else "Checking. Kingsway moved to Thursday."
        )
        return [("m1", text)]

    user = FakeUser(["thursday"])
    bot = FakeBot(clock, {1: grows, 2: grows})
    record, _ = run(tmp_path, bot, user, clock)
    # Asked twice: about "Checking", then about the grown answer, which it
    # reads although it sits before its reply.
    assert user.seen[0][-1] == "Checking"
    assert "Checking. Kingsway moved to Thursday." in user.seen[1]
    assert record.result["end_reason"] == "assistant_finished"
    edited = [t for t in record.turns if t.role == "assistant"][0]
    assert edited.edited_at_ms is not None


def test_a_judge_failure_mid_run_is_a_judge_error_not_a_setup_error(tmp_path):
    from assistanteval.grader.judges import JudgeError

    class BrokenUser(FakeUser):
        def reply(self, turns, at_ms):
            raise JudgeError("reader: HTTP 503")

    clock = Clock()
    record, service = run(
        tmp_path, FakeBot(clock, {1: steady(("m1", "done"))}), BrokenUser(), clock
    )
    assert record.result["end_reason"] == "judge_error"
    assert record.result["opening_sent"] is True and service.released


def test_at_the_turn_limit_the_user_can_still_end_but_a_reply_is_not_sent(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run_task, "MAX_USER_TURNS", 1)

    class Recording(FakeUser):
        def reply(self, turns, at_ms):
            self.last = super().reply(turns, at_ms)
            return self.last

    clock = Clock()
    finished = Recording()  # "end:finished" on the last allowed turn
    record, _ = run(
        tmp_path, FakeBot(clock, {1: steady(("m1", "Done."))}), finished, clock
    )
    assert record.result["end_reason"] == "assistant_finished"

    clock = Clock()
    more = Recording(["one more thing"])
    record, _ = run(tmp_path, FakeBot(clock, {1: steady(("m1", "hi"))}), more, clock)
    assert record.result["end_reason"] == "turn_limit"
    # The reply was decided but never sent, and says so for the grader.
    assert more.last.decision["sent"] is False
    assert [t.text for t in record.turns if t.role == "user"] == [
        "move kingsway to thursday"
    ]


def test_a_nudge_is_not_a_reply_so_a_later_rewrite_is_shown(tmp_path):
    clock = Clock()
    working = [("m1", "Working on it…")]
    # After the nudge, the product rewrites its first message into the answer.
    bot = FakeBot(clock, {1: working, 2: [("m1", "Kingsway moved to Thursday.")]})
    user = FakeUser(["wait"])
    record, _ = run(tmp_path, bot, user, clock)
    assert bot.sent[1:] == ["Any update?"]
    assert record.result["end_reason"] == "assistant_finished"
    assert user.seen[-1][-2:] == ["Kingsway moved to Thursday.", "Any update?"]


def test_a_service_whose_start_state_cannot_be_read_is_still_released(tmp_path):
    class Unreadable(FakeService):
        def state(self):
            raise EpisodeStepError("calendar_state_unreadable")

    clock = Clock()
    record, service = run(
        tmp_path, FakeBot(clock), FakeUser(), clock, service=Unreadable()
    )
    assert record.result["error"] == "calendar_state_unreadable"
    assert service.released  # its key is revoked, not left to expire
    assert record.result["cleanup_errors"] == {}


@pytest.mark.parametrize(
    "error, opening_sent",
    [
        # Never delivered: a retry is safe.
        ("prompt_not_sent", False),
        # The assistant may have seen the task: a retry would not be a fresh run.
        ("prompt_delivery_unconfirmed", True),
    ],
)
def test_an_opening_is_retried_only_if_it_never_arrived(tmp_path, error, opening_sent):
    class Unsent(FakeBot):
        def send(self, text):
            raise EpisodeStepError(error)

    clock = Clock()
    record, service = run(tmp_path, Unsent(clock), FakeUser(), clock)
    assert record.result["error"] == error
    assert record.result["opening_sent"] is opening_sent
    # Recorded only if the assistant may have seen it.
    assert [t.text for t in record.turns if t.role == "user"] == [
        "move kingsway to thursday"
    ] * opening_sent
    assert service.released


def test_an_installed_runner_never_asks_git(monkeypatch, tmp_path):
    """Installed (no .git at its root), git would name the repository the
    package happens to be installed in."""

    def git(*args, **kwargs):
        raise AssertionError("asked git")

    monkeypatch.setattr(run_task, "ROOT", tmp_path)
    monkeypatch.setattr(run_task.subprocess, "run", git)
    monkeypatch.setenv("ASSISTANTEVAL_REVISION", "abc123")
    assert run_task._runner_version() == "assistanteval.run_task at abc123"
