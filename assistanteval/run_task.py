"""Run one task end to end and save its run record.

    uv run python -m assistanteval.run_task tasks/<id>.json --assistant model_baseline

The runner owns the order of steps, the clock and the limits. The assistant
adapter drives the assistant, the simulated user decides what to say, and each
service keeps its own state and request log; they never call each other. The
record is written in the format of ``docs/run-record.md``, and grading happens
later, from the record alone.

The task's world is anchored to the run's start: the task and its fixtures
are rendered then (``assistanteval.dates``), and the record keeps the rendered
task and the anchor.

``--assistant`` names a built-in assistant, or any factory as
``module:attribute``, so an adapter can live in its own package.

Exit codes: 0 the run finished, 1 setup_error (see run.json), 6 judge_error
(the simulated user's judge failed mid-run).

One alarm stops the whole run at RUN_DEADLINE_SECONDS.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from assistanteval import connectors, dates, endpoints, run_record
from assistanteval.contracts import Assistant, EpisodeStepError, Message
from assistanteval.grader.judges import JudgeError
from assistanteval.run_record import RunRecord, Service, Turn
from assistanteval.task import load as load_task

ROOT = Path(__file__).resolve().parents[1]

# Limits, copied into every record's manifest.
POLL_SECONDS = 1
SETTLE_SECONDS = 5  # nothing new for this long: the user replies
SILENCE_SECONDS = 180  # nothing new for this long: nudge
MAX_NUDGES = 3  # unanswered nudges in a row: timeout
MAX_USER_TURNS = 30
DEADLINE_SECONDS = 45 * 60
# The whole run, setup and cleanup included.
RUN_DEADLINE_SECONDS = 85 * 60

# Assistant name -> factory. A factory opens a started `Assistant`.
ASSISTANTS = {
    "model_baseline": "assistanteval.assistants.model_baseline:open_chat",
    "reference": "assistanteval.assistants.reference:open_chat",
}


class Stop(Exception):
    """A stop signal or the run's deadline: save the record and release."""

    def __init__(self, code: str = "stopped"):
        super().__init__(code)
        self.code = code


def _resolve(entry: str):
    module, _, name = entry.partition(":")
    return getattr(importlib.import_module(module), name)


def _now_ms() -> int:
    return int(time.time() * 1000)


def simulated_user(task: dict):
    from assistanteval.endpoints import endpoint, required
    from assistanteval.grader.judges import judge_for
    from assistanteval.simulated_user import LLMWriter, SimulatedUser

    judge = judge_for(required("JUDGE"))
    writer = endpoint("WRITER")
    return SimulatedUser(
        task,
        judge_for(endpoint("READER")) or judge,
        writer and LLMWriter(writer),
        rules=judge,
        style=os.environ.get("ASSISTANTEVAL_USER_STYLE") or None,
    )


def converse(
    bot: Assistant,
    user,
    turns: list[Turn],
    clicks: list[dict],
    *,
    now_ms: Callable[[], int] = _now_ms,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """The conversation, from the opening message to its end reason.

    A message counts once two reads in a row show the same text, so the user
    never sees a half-written message. A message the assistant rewrites is
    updated in place and shown to the user again. Assistants often send several
    messages in a row, so the user replies only once nothing new has appeared
    for SETTLE_SECONDS, and never while the adapter says its answer was still
    coming when it read them (``answering``, set by ``read()``): a model's "let
    me find Ben's details" is not its answer.
    """

    def say(text: str, nudge: bool = False) -> None:
        at = now_ms()  # stamped just before it is sent
        try:
            bot.send(text)
        except EpisodeStepError as error:
            # Sent but not confirmed: the assistant may have it, so it is
            # recorded and the task is not retried as a fresh run.
            if str(error) == "prompt_delivery_unconfirmed":
                turns.append(Turn("user", text, at, nudge))
            raise
        # Otherwise recorded only once delivered: a message that never
        # reached the assistant is not part of the conversation.
        turns.append(Turn("user", text, at, nudge))
        _show("USER (nudge)" if nudge else "USER", text)

    def user_turns() -> int:
        return sum(turn.role == "user" for turn in turns)

    say(user.opening())
    started = quiet_since = now_ms()
    index: dict[str, int] = {}  # message id -> its turn
    unsteady: dict[str, str] = {}  # message id -> text seen once
    nudges = 0
    unseen = False  # messages the user has not replied to yet
    while now_ms() - started < DEADLINE_SECONDS * 1000:
        messages, answered = bot.read()
        # Set by read() with these messages, so it describes exactly them.
        answering = getattr(bot, "answering", False)
        clicks.extend({"at_ms": now_ms(), **click} for click in answered)
        if _absorb(messages, turns, index, unsteady, now_ms()):
            unseen, nudges, quiet_since = True, 0, now_ms()
        elif unsteady:
            quiet_since = now_ms()  # still writing: not silence
        if not unseen:
            if now_ms() - quiet_since >= SILENCE_SECONDS * 1000:
                if nudges == MAX_NUDGES:
                    return "timeout"
                if user_turns() >= MAX_USER_TURNS:
                    return "turn_limit"
                nudges += 1
                say(user.nudge(len(turns) - 1, now_ms()).text, nudge=True)
                quiet_since = now_ms()
            sleep(POLL_SECONDS)
            continue
        if unsteady or now_ms() - quiet_since < SETTLE_SECONDS * 1000 or answering:
            sleep(POLL_SECONDS)  # more may be on its way
            continue
        unseen, quiet_since = False, now_ms()
        reply = user.reply(turns, now_ms())
        _show("  user", f"{reply.decision.get('saw')} -> {reply.action}")
        if reply.action == "end":
            finished = reply.decision.get("saw") == "finished"
            return "assistant_finished" if finished else "user_ended"
        if reply.action == "wait":
            sleep(POLL_SECONDS)
            continue
        if user_turns() >= MAX_USER_TURNS:
            # The reply is never sent, so it must not count as said.
            reply.decision["sent"] = False
            return "turn_limit"
        say(reply.text)
    return "timeout"


def _show(who: str, text: str) -> None:
    """One progress line, so a live run can be followed in its log."""
    print(f"{who}: {' '.join(text.split())[:300]}", flush=True)


def _absorb(
    messages: list[Message],
    turns: list[Turn],
    index: dict[str, int],
    unsteady: dict[str, str],
    now: int,
) -> bool:
    """Record each message that two reads in a row show the same: a new
    turn, or its turn's new text. True if any was recorded."""
    recorded = False
    for message in messages:
        known = turns[index[message.id]].text if message.id in index else None
        if not message.text or message.text == known:
            unsteady.pop(message.id, None)
        elif unsteady.get(message.id) != message.text:
            unsteady[message.id] = message.text
        else:
            del unsteady[message.id]
            recorded = True
            if message.id in index:
                # First shown stays its time; the edit gets its own. The user
                # reads it again with its new text, even if it had already
                # replied to it: an assistant may grow an answer in place.
                turn = turns[index[message.id]]
                turn.text, turn.edited_at_ms = message.text, now
                _show("ASSISTANT (edited)", message.text)
            else:
                index[message.id] = len(turns)
                turns.append(Turn("assistant", message.text, now))
                _show("ASSISTANT", message.text)
    shown = {message.id for message in messages}
    for gone in set(unsteady) - shown:  # a half-written message since replaced
        del unsteady[gone]
    return recorded


def _code(error: BaseException) -> str:
    # Messages can carry private text or keys; keep only a fixed code.
    return str(error) if isinstance(error, EpisodeStepError) else type(error).__name__


def _runner_version() -> str:
    # A checkout knows its commit; an installed package records it.
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:  # no git where the runner runs
        revision = ""
    revision = revision or os.environ.get("ASSISTANTEVAL_REVISION") or "unknown"
    return f"assistanteval.run_task at {revision}"


def run_task(
    task: dict,
    assistant: str,
    open_assistant,
    out_dir: Path,
    *,
    user=None,
    services_from: Callable = connectors.open_service,
    now_ms: Callable[[], int] = _now_ms,
    sleep: Callable[[float], None] = time.sleep,
    manifest: dict | None = None,
    root: Path = ROOT,
) -> RunRecord:
    """Run `task` on one assistant and save its record in `out_dir`.

    The task's world (the task and its fixtures) is rendered at the run's
    start, so its dates are current whenever it runs; the simulated user and
    the services see only the rendered world. Its fixture paths are relative
    to ``root``. The record's manifest is ``manifest`` (the caller's, such as
    the task file's hash), the runner's version, the model endpoints, the
    factory's ``manifest()`` if it has one, and what this run used.

    Anything that fails ends the run as ``setup_error``, which is never
    scored; ``result.opening_sent`` says whether a retry is safe. Once the
    conversation has ended, a failure closing the assistant or releasing a
    service is only a ``cleanup_errors`` entry, as is a failure closing the
    assistant after the run had already failed.
    ``result.account_ok`` is False only when the assistant itself refused to
    start or reset (an `EpisodeStepError` while opening or resetting). A
    network or code failure there makes it None: unverified.
    """
    common = {
        **(manifest or {}),
        "runner_version": _runner_version(),
        # The models besides the assistant, without their keys.
        "endpoints": endpoints.recorded(),
        # What the assistant is, when its name alone does not say (a model id).
        **getattr(open_assistant, "manifest", dict)(),
    }
    # Before the assistant starts, so its local clock is the task's.
    os.environ["TZ"] = task["environment"]["timezone"]
    time.tzset()
    turns, clicks, setup, services = [], [], [], {}
    rendered, scratch, failure = None, None, None
    result = {
        "end_reason": "setup_error",
        "reset": {"verified": False, "method": "not_reached"},
        "cleanup_errors": {},
        "opening_sent": False,
        "account_ok": True,
        "started_at_ms": now_ms(),
    }
    phase = "world"
    try:
        # Anchored here, at the run's start: the world is current when the
        # conversation starts.
        anchor = dates.anchor_at(now_ms(), task["environment"]["timezone"])
        rendered = dates.render(task, root, anchor)
        scratch = Path(tempfile.mkdtemp(prefix="assistanteval-world-"))
        phase = "user"
        user = user or simulated_user(rendered.task)
        phase = "services"
        for connector in task["environment"]["connectors"]:
            name = connector["service"]
            fixture = scratch / f"{name}.json"
            fixture.write_bytes(rendered.fixtures[name])
            service = services_from(name, fixture, open_assistant.connector_kind)
            # Registered before preparing, so a session that opens but then
            # fails (e.g. reading its start state) is still revoked below.
            services[name] = (service, None, None)
            prepared = service.prepare()
            services[name] = (service, prepared.connector, service.state())
        phase = "sign_in"
        with open_assistant(None, out_dir / "evidence") as bot:
            try:
                phase = "reset"
                print("run: reset", flush=True)
                proof = bot.reset()
                result["reset"] = dataclasses.asdict(proof)
                if not proof.verified:
                    raise EpisodeStepError("reset_not_verified")
                phase = "connect"
                print("run: connect", flush=True)
                setup += [
                    {
                        "role": "user",
                        "text": f"connector setup: {name}",
                        "at_ms": now_ms(),
                    }
                    for name in services
                ]
                setup += [
                    {"role": "assistant", "text": text, "at_ms": now_ms()}
                    for text in bot.connect([c for _, c, _ in services.values()])
                ]
                phase = "converse"
                print("run: converse", flush=True)
                result["end_reason"] = converse(
                    bot, user, turns, clicks, now_ms=now_ms, sleep=sleep
                )
            except BaseException as error:
                failure = error  # what the adapter's cleanup may replace
                raise
            phase = "close"
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - recorded as a code
        if failure is not None and error is not failure:
            # Closing the assistant failed while the run was already failing:
            # the run's own failure is its error.
            result["cleanup_errors"]["assistant"] = _code(error)
            error = failure
        _fail(result, phase, error)
    finally:
        result["phase"] = phase
        result["ended_at_ms"] = now_ms()
        # Sent means delivered: say() records a user turn only after send().
        result["opening_sent"] = any(turn.role == "user" for turn in turns)
        recorded = _release(services, result)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        user_record = user.record() if user else None
        record = RunRecord(
            out_dir.name,
            task["id"],
            assistant,
            turns,
            recorded,
            result,
            permission_clicks=clicks,
            simulated_user=user_record,
            setup=setup,
            # The task as the run used it: rendered at its anchor.
            task=rendered.task if rendered else task,
            manifest=_manifest(
                common, user_record, rendered, task["environment"]["timezone"]
            ),
        )
        run_record.save(record, out_dir)
    return record


def _fail(result: dict, phase: str, error: BaseException) -> None:
    """Record why the run failed in `phase`."""
    if isinstance(error, (Stop, KeyboardInterrupt)):
        result["error"] = getattr(error, "code", "stopped")
    elif isinstance(error, JudgeError):
        # The simulated user's judge failed mid-run: the harness's fault, not
        # a setup failure and not the assistant's.
        result["end_reason"] = "judge_error"
        result["error"] = _code(error)
    elif phase == "close":
        # The conversation stands; only the assistant's own cleanup (e.g.
        # removing this run's connectors) failed.
        result["cleanup_errors"]["assistant"] = _code(error)
    else:
        result["error"] = _code(error)
        if phase in ("sign_in", "reset"):
            # The assistant's own refusal (an EpisodeStepError: it would not
            # open, or a reset never cleared) makes account_ok False. A network
            # or code error leaves it unverified: None.
            refused = isinstance(error, EpisodeStepError)
            result["account_ok"] = False if refused else None


def _release(services: dict, result: dict) -> list[Service]:
    """Each read service's record, collected before every service is
    released; a failure doing either is a cleanup error."""
    recorded = []
    for name, (service, _, start) in services.items():
        if start is not None:  # nothing to collect from a session never read
            try:
                requests, _ = service.request_log()
                recorded.append(Service(name, start, service.state(), requests))
            except Exception as error:  # noqa: BLE001
                result["cleanup_errors"][f"collect_{name}"] = _code(error)
        try:
            service.release()
        except Exception as error:  # noqa: BLE001
            result["cleanup_errors"][name] = _code(error)
    return recorded


def _manifest(common: dict, user_record: dict | None, rendered, timezone: str) -> dict:
    """The record's manifest: `common`, then what this run used."""
    manifest = dict(common)
    if user_record is not None:
        # The simulated user's writer model, as a run's provenance.
        manifest["simulated_user"] = {"model": user_record.get("model")}
    manifest["browser_timezone"] = timezone
    if rendered:
        # The anchor the world was rendered at, and each rendered fixture
        # with its hash: the grader checks the run started while that world
        # was current, and each service from it.
        fixture_data = {
            name: json.loads(raw) for name, raw in rendered.fixtures.items()
        }
        manifest["world"] = rendered.manifest() | {"fixture_data": fixture_data}
    manifest["limits"] = {
        "poll_seconds": POLL_SECONDS,
        "settle_seconds": SETTLE_SECONDS,
        "silence_seconds": SILENCE_SECONDS,
        "max_nudges": MAX_NUDGES,
        "max_user_turns": MAX_USER_TURNS,
        "deadline_seconds": DEADLINE_SECONDS,
    }
    return manifest


@contextmanager
def _stop_on_signals():
    """Stop on SIGTERM or SIGHUP, and at the run's hard deadline (one alarm)."""

    def stop(signum, _frame):
        raise Stop("run_deadline" if signum == signal.SIGALRM else "stopped")

    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)
    previous = {s: signal.signal(s, stop) for s in signals}
    signal.setitimer(signal.ITIMER_REAL, RUN_DEADLINE_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", type=Path)
    parser.add_argument(
        "--assistant",
        required=True,
        help=f"one of {', '.join(sorted(ASSISTANTS))}, or module:factory",
    )
    parser.add_argument(
        "--out", type=Path, help="run folder (default runs/<assistant>-<task>-<time>)"
    )
    args = parser.parse_args(argv)

    task_bytes = args.task.read_bytes()
    task = load_task(args.task).dump()  # checked; rendered at the run's start
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = args.assistant.rpartition(":")[2]
    out_dir = args.out or Path("runs") / f"{name}-{task['id']}-{stamp}"
    factory = _resolve(ASSISTANTS.get(args.assistant, args.assistant))
    with _stop_on_signals():
        record = run_task(
            task,
            args.assistant,
            factory,
            out_dir,
            manifest={"task_sha256": hashlib.sha256(task_bytes).hexdigest()},
            root=args.task.resolve().parents[1],  # fixture paths are relative to it
        )
    reason = record.result["end_reason"]
    print(f"{reason}: {out_dir}")
    return {"setup_error": 1, "judge_error": 6}.get(reason, 0)


if __name__ == "__main__":
    raise SystemExit(main())
