"""Every model besides the assistant is set per role, on any OpenAI-compatible API."""

import json
from types import SimpleNamespace

import pytest

from assistanteval import endpoints, run_task
from assistanteval.endpoints import Endpoint, endpoint
from assistanteval.grader import judges
from assistanteval.grader.judges import (
    AnswerStore,
    JevJudge,
    JudgeError,
    LLMJudge,
    Question,
    judge_for,
    ladder_from_env,
)


@pytest.fixture(autouse=True)
def no_roles(monkeypatch):
    for role in endpoints.ROLES:
        for name in endpoints.SETTINGS:
            monkeypatch.delenv(f"ASSISTANTEVAL_{role}_{name}", raising=False)


def test_a_role_reads_its_own_settings(monkeypatch):
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_MODEL", "llama-4-judge")
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_API_KEY", "k")
    monkeypatch.setenv(
        "ASSISTANTEVAL_JUDGE_EXTRA_BODY", '{"reasoning": {"effort": "low"}}'
    )
    found = endpoint("JUDGE")
    assert found == Endpoint(
        "llama-4-judge",
        "http://localhost:11434/v1",
        "k",
        "openai",
        {"reasoning": {"effort": "low"}},
    )
    assert endpoint("READER") is None
    # Records keep the endpoint, never its key.
    assert "k" not in json.dumps(found.describe()).replace("llama-4-judge", "")
    assert "k" not in repr(found).replace("llama-4-judge", "")


def test_jev_is_only_a_reader(monkeypatch):
    monkeypatch.setenv("ASSISTANTEVAL_READER_MODEL", "jev-1.13.0")
    monkeypatch.setenv("ASSISTANTEVAL_READER_API", "jev")
    monkeypatch.setenv("ASSISTANTEVAL_READER_API_KEY", "k")
    assert isinstance(judge_for(endpoint("READER")), JevJudge)
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_MODEL", "jev-1.13.0")
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_API", "jev")
    with pytest.raises(ValueError, match="READER only"):
        endpoint("JUDGE")


def test_live_grading_and_runs_need_a_judge(monkeypatch):
    with pytest.raises(ValueError, match="ASSISTANTEVAL_JUDGE_MODEL"):
        ladder_from_env(AnswerStore())
    with pytest.raises(ValueError, match="ASSISTANTEVAL_JUDGE_MODEL"):
        run_task.simulated_user({"simulated_user": {"facts": []}})
    offline = ladder_from_env(AnswerStore(), offline=True)
    assert offline.reader is offline.judge is None


def test_without_a_reader_the_judge_reads_for_the_simulated_user(monkeypatch):
    monkeypatch.setenv("ASSISTANTEVAL_JUDGE_MODEL", "judge-model")
    user = run_task.simulated_user({"id": "t", "simulated_user": {"facts": []}})
    assert user.reader is user.rules and user.reader.model == "judge-model"
    assert user.writer is None  # the fixed replies are sent as they are
    monkeypatch.setenv("ASSISTANTEVAL_WRITER_MODEL", "writer-model")
    monkeypatch.setenv("ASSISTANTEVAL_READER_MODEL", "reader-model")
    user = run_task.simulated_user({"id": "t", "simulated_user": {"facts": []}})
    assert (user.reader.model, user.rules.model, user.writer.model) == (
        "reader-model",
        "judge-model",
        "writer-model",
    )


class FakeClient:
    """An OpenAI client that replies with `content` and keeps each request."""

    def __init__(self, *contents):
        self.requests = []
        self.chat = SimpleNamespace(completions=self)
        self.contents = list(contents)

    def create(self, **request):
        self.requests.append(request)
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize(
    "content",
    ['{"p_yes": 0.9}', '```json\n{"p_yes": 0.9}\n```', 'Answer: {"p_yes": 0.9}'],
)
def test_the_llm_judge_reads_json_from_models_without_a_json_mode(monkeypatch, content):
    client = FakeClient(content)
    monkeypatch.setattr(Endpoint, "client", lambda self, **options: client)
    judge = LLMJudge(Endpoint("m", extra_body={"top_k": 1}))
    assert judge(Question("q", "yes_no", "Is it?", {})) == 0.9
    assert client.requests[0]["model"] == "m"
    assert client.requests[0]["extra_body"] == {"top_k": 1}


def test_an_llm_reply_without_an_answer_is_a_judge_error(monkeypatch):
    monkeypatch.setattr(
        Endpoint, "client", lambda self, **options: FakeClient("no idea")
    )
    with pytest.raises(JudgeError):
        LLMJudge(Endpoint("m"))(Question("q", "yes_no", "Is it?", {}))


def test_the_verdict_names_the_models_that_judged(monkeypatch):
    ladder = judges.Ladder(
        AnswerStore(),
        reader=SimpleNamespace(model="reader-model"),
        judge=SimpleNamespace(model="judge-model"),
    )
    assert ladder.models() == {"reader": "reader-model", "judge": "judge-model"}


def test_a_malformed_reply_is_asked_again(monkeypatch):
    options = {"supported": "...", "none of these": "..."}
    q = Question("q", "pick_one", "Which?", {}, criteria=options)
    client = FakeClient(
        '{"verdict": "supported"}', '{"supported": 0.9, "none of these": 0.1}'
    )
    monkeypatch.setattr(Endpoint, "client", lambda self, **options: client)
    # The second reply leaves out the "probabilities" wrapper but names the options.
    assert LLMJudge(Endpoint("m"))(q) == {"supported": 0.9, "none of these": 0.1}
    assert len(client.requests) == 2


def test_a_judge_is_known_by_its_model_server_and_options(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert Endpoint("gpt-5.6-sol").identity == "gpt-5.6-sol"  # the default server
    routed = Endpoint("openai/gpt-5.6-sol", "https://openrouter.ai/api/v1")
    assert routed.identity == "openai/gpt-5.6-sol @ https://openrouter.ai/api/v1"
    tuned = Endpoint("m", "http://localhost:8000/v1", extra_body={"top_k": 1})
    assert tuned.identity == 'm @ http://localhost:8000/v1 {"top_k": 1}'
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy/v1")
    assert Endpoint("gpt-5.6-sol").describe()["base_url"] == "http://proxy/v1"


def test_saved_answers_are_reused_only_by_the_same_judge(monkeypatch):
    """The same model name on another server is another judge."""
    clients = {
        "http://old/v1": FakeClient('{"p_yes": 0.9}'),
        "http://new/v1": FakeClient('{"p_yes": 0.1}'),
    }
    monkeypatch.setattr(
        Endpoint, "client", lambda self, **options: clients[self.base_url]
    )
    q = Question("q", "yes_no", "Is it?", {})
    store = AnswerStore()
    old = judges.Ladder(store, judge=LLMJudge(Endpoint("judge", "http://old/v1")))
    assert old.decide_one(q).value is True
    new = judges.Ladder(store, judge=LLMJudge(Endpoint("judge", "http://new/v1")))
    assert new.decide_one(q).value is False
    assert len(clients["http://new/v1"].requests) == 1


@pytest.mark.parametrize(
    "content, kind",
    [
        ('{"p_yes": 90}', "yes_no"),
        ('{"p_yes": NaN}', "yes_no"),
        ('{"a": 9, "b": 8}', "pick_one"),
        ('{"a": 0.9, "b": 0.8}', "pick_one"),
    ],
)
def test_numbers_that_are_not_probabilities_are_asked_again(monkeypatch, content, kind):
    client = FakeClient(content)
    monkeypatch.setattr(Endpoint, "client", lambda self, **options: client)
    q = Question(
        "q",
        kind,
        "Which?",
        {},
        criteria={"a": "", "b": ""} if kind == "pick_one" else None,
    )
    with pytest.raises(JudgeError):
        LLMJudge(Endpoint("m"))(q)
    assert len(client.requests) == judges.MALFORMED_TRIES
