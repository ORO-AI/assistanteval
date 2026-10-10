"""The model-baseline assistant, offline: a scripted model against real local services."""

import hashlib
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from assistanteval import connectors, run_task
from assistanteval.contracts import EpisodeStepError
from assistanteval.assistants import model_baseline
from assistanteval.assistants.model_baseline import (
    ModelBaselineChat,
    open_chat,
    openapi_tools,
)
from tests.conftest import SERVICES_URL, in_process_services

TASK = {
    "id": "model-baseline-demo",
    "environment": {
        "connectors": [
            {"service": "marketplace", "fixture": "tests/fixtures/marketplace.json"}
        ],
        "timezone": "America/Vancouver",
    },
}


@pytest.fixture
def local_marketplace(tmp_path, monkeypatch):
    """The marketplace service on a local store, opened by `open_service`, and
    an HTTP client that reaches its REST API."""
    services_from, client = in_process_services(tmp_path / "sessions")
    monkeypatch.setattr(connectors, "open_service", services_from)
    client.base_url = f"{SERVICES_URL}/marketplace"
    return client


def call(id_, name, **arguments):
    return {
        "id": id_,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedModel:
    """Answers each model call from `script`, in order, and remembers what
    it was shown."""

    def __init__(self, *script):
        self.script, self.seen = list(script), []

    def __call__(self, messages, tools):
        self.seen.append((messages, tools))
        step = self.script.pop(0) if self.script else {"content": "Done."}
        if callable(step):
            step = step(messages, tools)
        return dict(step), {"prompt_tokens": 10, "completion_tokens": 2}


def replying_model():
    return ScriptedModel(
        {
            "content": "Let me check that thread.",
            "tool_calls": [
                call("c1", "marketplace__read_listings", thread_id="thr-012")
            ],
        },
        {
            "tool_calls": [
                call(
                    "c2",
                    "marketplace__reply_to_thread",
                    thread_id="thr-012",
                    message="Yes, the desk chair is still available.",
                )
            ]
        },
        {"content": "I told Jordan the desk chair is still available."},
    )


def until(predicate, seconds=5):
    """Polls `predicate` until it returns something true, and returns that."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.01)
    raise AssertionError(f"not true after {seconds}s")


def connector():
    service = connectors.open_service(
        "marketplace",
        run_task.ROOT / TASK["environment"]["connectors"][0]["fixture"],
        "openapi_api_key",
    )
    return service, service.prepare().connector


def test_openapi_operations_become_tools_named_after_the_operation(local_marketplace):
    schema = local_marketplace.get("/openapi.json").json()
    tools = {tool["name"]: tool for tool in openapi_tools("Marketplace", schema)}
    assert set(tools) == {
        "marketplace__read_listings",
        "marketplace__reply_to_thread",
    }
    reply = tools["marketplace__reply_to_thread"]
    assert reply["parameters"]["required"] == ["thread_id", "message"]
    assert reply["route"] == (
        "POST",
        "/v1/threads/{thread_id}/messages",
        {"thread_id": "path", "message": "body"},
    )
    assert reply["description"].startswith("Send a message to a buyer")
    assert "$ref" not in json.dumps(tools)


def test_the_loop_calls_the_service_and_reports_each_text_once(
    local_marketplace, tmp_path
):
    service, key = connector()
    model = replying_model()
    with open_chat(
        None, tmp_path / "evidence", model=model, http=local_marketplace
    ) as chat:
        assert chat.reset().verified
        assert chat.connect([key]) == []
        chat.send("Can you tell Jordan the desk chair is still available?")
        until(lambda: len(chat.read()[0]) >= 2)
        messages, _ = chat.read()
        assert [(m.id, m.text) for m in messages] == [
            ("0", "Let me check that thread."),
            ("1", "I told Jordan the desk chair is still available."),
        ]
        assert chat.read() == (messages, [])  # no edits, no permission prompts
    requests, complete = service.request_log()
    assert complete
    assert [(r.operation, r.input) for r in requests] == [
        ("marketplace.read", {"thread_id": "thr-012"}),
        (
            "marketplace.reply",
            {
                "thread_id": "thr-012",
                "message": "Yes, the desk chair is still available.",
            },
        ),
    ]
    # The model saw the service's own answer, and never the key.
    messages, tools = model.seen[1]
    result = json.loads(messages[-1]["content"])
    assert result["status"] == 200 and "Jordan Lee" in result["body"]
    assert key.key not in json.dumps(model.seen)
    trace = json.loads((tmp_path / "evidence/model_baseline_trace.json").read_text())
    assert trace["steps"] == 3 and trace["usage"]["prompt_tokens"] == 30
    assert key.key not in json.dumps(trace)
    service.release()


NEUTRAL = (
    "You are the user's personal assistant. You can act on their accounts "
    "through the tools provided."
)


def test_models_get_the_neutral_prompt_and_the_manifest_names_it(monkeypatch):
    """The role and the tools only, as a product starts: a prompt that hints at
    what is graded would make model results incomparable with products'."""
    model = ScriptedModel({"content": "Hi."})
    chat = ModelBaselineChat(
        model, http=None, now=lambda: datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc)
    )
    chat.send("hello")
    until(lambda: chat.read()[0])
    chat.close()
    system = model.seen[0][0][0]
    assert system["role"] == "system"
    assert system["content"].split("\n\n")[0] == NEUTRAL
    # It gives the user's local time.
    assert "Thursday, October 1, 2026, 9:30 AM (UTC)" in system["content"]
    monkeypatch.setenv("ASSISTANTEVAL_BASELINE_MODEL", "anthropic/claude-opus-5.5")
    recorded = model_baseline.manifest()["model_baseline"]
    assert recorded["system_prompt"] == "neutral"
    assert (
        recorded["system_prompt_sha256"] == hashlib.sha256(NEUTRAL.encode()).hexdigest()
    )


def test_a_message_sent_while_the_model_works_joins_its_next_call():
    chat = None

    def slow_first(messages, tools):
        chat.send("Also, keep it short.")  # the user speaks again mid-turn
        return {"content": "Working on it.", "tool_calls": [call("c1", "nope")]}

    model = ScriptedModel(slow_first, {"content": "Short answer."})
    chat = ModelBaselineChat(model, http=None)
    chat.send("Help me")
    until(lambda: len(chat.read()[0]) >= 2)
    assert [m.text for m in chat.read()[0]] == ["Working on it.", "Short answer."]
    chat.close()
    second = model.seen[1][0]
    assert json.loads(second[-2]["content"]) == {"error": "No tool named nope"}
    assert second[-1] == {"role": "user", "content": "Also, keep it short."}


def test_the_chat_is_answering_until_the_model_answers_without_a_tool_call():
    chat, during = None, []

    def first(messages, tools):
        chat.read()  # answering is set by read(), with its messages
        during.append(chat.answering)
        return {"content": "Let me look.", "tool_calls": [call("c1", "nope")]}

    chat = ModelBaselineChat(ScriptedModel(first, {"content": "Done."}), http=None)
    chat.send("Help me")

    def answered():
        messages, _ = chat.read()
        return not chat.answering and messages

    # Once it is not answering, the same read holds its answer.
    assert [m.text for m in until(answered)][-1] == "Done."
    assert during == [True]
    chat.close()


def test_bad_tool_arguments_go_back_to_the_model(local_marketplace):
    service, key = connector()
    model = ScriptedModel(
        {"tool_calls": [call("c1", "marketplace__reply_to_thread", message="hi")]},
        {
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "marketplace__read_listings",
                        "arguments": "{",
                    },
                }
            ]
        },
        {
            "tool_calls": [
                call(
                    "c3",
                    "marketplace__reply_to_thread",
                    thread_id="thr-012",
                    message="",
                )
            ]
        },
        {"content": "I could not send it."},
    )
    chat = ModelBaselineChat(model, local_marketplace)
    chat.connect([key])
    chat.send("Reply hi")
    until(lambda: chat.read()[0])
    chat.close()
    results = [json.loads(m["content"]) for m in chat.history if m["role"] == "tool"]
    assert results[0] == {"error": "Missing required argument thread_id"}
    assert results[1] == {"error": "Arguments are not valid JSON"}
    assert results[2]["status"] == 422  # the service refused an empty message
    service.release()


def test_a_model_that_never_stops_calling_tools_goes_quiet(monkeypatch):
    monkeypatch.setattr(model_baseline, "MAX_STEPS", 3)
    looping = {"tool_calls": [call("c", "nope")]}
    model = ScriptedModel(*[looping] * 10)
    chat = ModelBaselineChat(model, http=None)
    chat.send("go")
    until(lambda: chat.step_limits)
    assert chat.step_limits == 1 and chat.steps == 3
    assert chat.read() == ([], [])  # not an error: the runner nudges
    chat.send("Any update?")  # and a nudge starts a new loop
    until(lambda: chat.step_limits >= 2)
    assert chat.steps == 6
    chat.close()


def test_a_failed_model_call_is_a_fixed_code_without_its_text():
    def broken(messages, tools):
        raise RuntimeError("401 for key sk-or-secret")

    chat = ModelBaselineChat(broken, http=None)
    chat.send("hello")

    def failure():
        try:
            chat.read()
        except EpisodeStepError as error:
            return str(error)

    assert until(failure) == "model_loop_failed"
    with pytest.raises(EpisodeStepError, match="model_loop_failed"):
        chat.send("again")


def test_reset_is_verified_only_on_a_new_chat():
    chat = ModelBaselineChat(ScriptedModel(), http=None)
    assert chat.reset() == model_baseline.ResetEvidence(True, "new_chat_empty_history")
    chat.send("hi")
    until(lambda: chat.read()[0])
    chat.close()
    assert not chat.reset().verified


def test_only_rest_connectors_are_supported():
    chat = ModelBaselineChat(ScriptedModel(), http=None)
    mcp = SimpleNamespace(kind="mcp_bearer", name="Contacts", url="h", key="k")
    with pytest.raises(EpisodeStepError, match="connector_kind_unsupported"):
        chat.connect([mcp])


def test_the_model_and_its_endpoint_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("ASSISTANTEVAL_BASELINE_MODEL", raising=False)
    with pytest.raises(ValueError, match="ASSISTANTEVAL_BASELINE_MODEL"):
        model_baseline.manifest()
    monkeypatch.setenv("ASSISTANTEVAL_BASELINE_MODEL", "qwen3.6-35b")
    monkeypatch.setenv("ASSISTANTEVAL_BASELINE_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("ASSISTANTEVAL_BASELINE_API_KEY", "secret-key")
    pinned = model_baseline.manifest()["model_baseline"]
    assert pinned["model"] == "qwen3.6-35b"
    assert pinned["base_url"] == "http://localhost:8000/v1"
    assert "secret-key" not in json.dumps(pinned)


def test_a_whole_run_leaves_a_gradeable_record(
    local_marketplace, tmp_path, monkeypatch
):
    """The runner, the real adapter and the real local service together."""
    monkeypatch.setattr(run_task, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(run_task, "SETTLE_SECONDS", 0.05)

    @contextmanager
    def factory(account, evidence_dir):
        with open_chat(
            account, evidence_dir, model=replying_model(), http=local_marketplace
        ) as chat:
            yield chat

    factory.connector_kind = "openapi_api_key"
    user = SimpleNamespace(
        opening=lambda: "Tell Jordan the desk chair is still available.",
        reply=lambda turns, at_ms: SimpleNamespace(
            action="end", text=None, decision={"saw": "finished"}
        ),
        record=lambda: {},
    )
    record = run_task.run_task(
        TASK,
        "model_baseline",
        factory,
        tmp_path / "run",
        user=user,
        services_from=connectors.open_service,
    )
    assert record.result["end_reason"] == "assistant_finished", record.result
    assert record.result["reset"] == {
        "verified": True,
        "method": "new_chat_empty_history",
        "isolated": True,
    }
    assert [t.text for t in record.turns if t.role == "assistant"] == [
        "Let me check that thread.",
        "I told Jordan the desk chair is still available.",
    ]
    (marketplace,) = record.services
    assert [r.operation for r in marketplace.requests] == [
        "marketplace.read",
        "marketplace.reply",
    ]
    assert (tmp_path / "run/evidence/model_baseline_trace.json").is_file()


def test_a_null_query_argument_is_omitted_not_sent_empty(local_marketplace):
    """Some models pass optional arguments as null; `thread_id=` would filter for
    thread "" and hide every listing."""
    service, key = connector()
    model = ScriptedModel(
        {"tool_calls": [call("c1", "marketplace__read_listings", thread_id=None)]},
        {"content": "Read."},
    )
    chat = ModelBaselineChat(model, local_marketplace)
    chat.connect([key])
    chat.send("What is listed?")
    until(lambda: chat.read()[0])
    chat.close()
    result = json.loads(json.loads(chat.history[2]["content"])["body"])
    assert result["listings"]
    service.release()
