"""The ACP adapter, over a fake acpx."""

import json
import threading
import time

import pytest

from assistanteval.assistants.acp import AcpChat, messages_of
from assistanteval.contracts import EpisodeStepError


def update(kind, **fields):
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"update": {"sessionUpdate": kind, **fields}},
        }
    )


def chunk(text):
    return update("agent_message_chunk", content={"type": "text", "text": text})


TURN = "\n".join(
    [
        json.dumps(
            {"jsonrpc": "2.0", "id": "1", "method": "session/prompt", "params": {}}
        ),
        chunk("Let me check "),
        chunk("your calendar."),
        update("tool_call", toolCallId="t1", title="calendar__search_events"),
        update("tool_call_update", toolCallId="t1", status="completed"),
        chunk("Kingsway is on Tuesday."),
        update("agent_thought_chunk", content={"type": "text", "text": "hidden"}),
        json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"stopReason": "end_turn"}}),
    ]
)


def test_a_turn_is_split_into_messages_at_its_tool_calls():
    texts, events = messages_of(TURN)
    assert texts == ["Let me check your calendar.", "Kingsway is on Tuesday."]
    assert len(events) == 8


def wait(chat, count):
    """The messages once `count` are shown and no turn runs; a failed run
    raises from `read()`."""
    for _ in range(500):
        messages, _ = chat.read()
        if len(messages) >= count and not chat.answering:
            return messages
        time.sleep(0.01)
    raise AssertionError(chat.read())


def test_each_message_is_one_prompt_and_messages_sent_mid_turn_wait():
    commands, release = [], threading.Event()

    def run(command):
        commands.append(command)
        if len(commands) == 1:
            release.wait(5)
        return 0, chunk(f"reply {len(commands)}")

    chat = AcpChat(run)
    assert chat.reset().verified
    chat.send("move kingsway to tuesday")
    chat.send("and tell doug")  # the first turn is still running
    assert chat.read()[0] == [] and chat.answering
    release.set()
    messages = wait(chat, 2)
    assert [(m.id, m.text) for m in messages] == [("0", "reply 1"), ("1", "reply 2")]
    assert commands == [
        "acpx --format json --json-strict prompt 'move kingsway to tuesday'",
        "acpx --format json --json-strict prompt 'and tell doug'",
    ]
    assert not chat.reset().verified  # no longer a new session


def test_each_prompt_runs_in_the_senders_context():
    """A host's per-task state, such as Harbor's scoped environment, reaches acpx."""
    import contextvars

    scope = contextvars.ContextVar("scope", default="none")
    seen = []
    chat = AcpChat(lambda command: (seen.append(scope.get()), (0, chunk("ok")))[1])
    scope.set("harness keys")
    chat.send("hi")
    wait(chat, 1)
    assert seen == ["harness keys"]


REFUSED = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"stopReason": "refusal"}})


@pytest.mark.parametrize(
    "run, error, kept",
    [
        (lambda command: (1, chunk("partial")), "acp_prompt_failed", ["partial"]),
        # Refused without a word: the run fails rather than the assistant.
        (lambda command: (0, REFUSED), "acp_prompt_refused", []),
        (
            lambda command: (_ for _ in ()).throw(RuntimeError("exec failed")),
            "acp_prompt_failed",
            [],
        ),
        (lambda command: (0, "not json\n"), "acp_prompt_failed", []),
    ],
    ids=["prompt fails", "refusal", "exec fails", "output not json"],
)
def test_a_failed_turn_fails_the_run_instead_of_hanging(run, error, kept):
    """Otherwise the thread dies while answering, and the run ends as a scored
    timeout instead of a setup error."""
    chat = AcpChat(run)
    chat.send("hello")
    with pytest.raises(EpisodeStepError, match=error):
        wait(chat, 1)
    # What a failed prompt wrote is kept in the evidence.
    texts = [
        e["params"]["update"]["content"]["text"] for e in chat.events if "params" in e
    ]
    assert texts == kept
