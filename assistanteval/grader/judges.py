"""The judge ladder for questions that code cannot answer.

Order: a saved human annotation, then the reader, then the judge, then the human annotation queue.
The reader and the judge are any models, set per role (see `assistanteval.endpoints`): an
OpenAI-compatible chat API, or TypeSafe's Jev for the reader.
Every answer is saved, keyed by the exact question and the rung that answered it, with the
model that gave it, so a saved run can be graded again with no API calls.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from assistanteval.endpoints import Endpoint, endpoint, required
from assistanteval.run_record import canonical, sha256

# A judge is sure when yes/no is <= LOW or >= HIGH, or one option has >= HIGH.
LOW, HIGH = 0.3, 0.7
# Earlier messages a judge sees with the assistant's turn: enough to hold a
# draft proposed several turns back. The simulated user and the grader's
# re-checks of its readings see the same.
CONTEXT_TURNS = 12
JEV_URL = "https://api.typesafe.ai/v1"
HUMAN = "human"
RETRY_CODES = {429, 500, 502, 503, 504, 529}
MALFORMED_TRIES = 3  # asks of a judge whose reply has no readable answer
JEV_TRIES = 6  # asks of Jev that fail to send or get a retryable error


class JudgeError(Exception):
    """A judge could not answer, even after retries. The question moves to the next step."""


@dataclass(frozen=True)
class Question:
    """One narrow question about text. kind is "yes_no" or "pick_one"."""

    id: str
    kind: str
    text: str
    context: dict
    criteria: dict | None = None
    # The rungs that may answer, lowest first. A re-check of an answer the
    # judge already gave leaves the judge out, so it is not asked to agree
    # with itself; when the reader is unsure, a person decides.
    rungs: tuple[str, ...] = ("reader", "judge")

    def key(self) -> str:
        """Same question text, criteria and context give the same key."""
        return sha256(canonical([self.kind, self.text, self.criteria, self.context]))[
            :32
        ]


@dataclass
class Decision:
    question_id: str
    value: bool | str | None  # None means it needs a human annotation
    decided_by: str  # "human", "reader", "judge", or "needs_human_annotation"
    reader: object = None
    judge: object = None
    note: str = ""
    checker: object = None  # set only when a CHECKER answered
    question: Question | None = None  # kept only when a human must answer it


def read(kind: str, answer: object) -> tuple[bool | str, bool]:
    """Return (decision, sure) for a probability or a distribution."""
    if kind == "yes_no":
        return answer >= 0.5, answer <= LOW or answer >= HIGH
    top = max(answer, key=answer.get)
    return top, answer[top] >= HIGH


class AnswerStore:
    """Saved answers: {"answers": {"<judge>:<question key>": {...}}}."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.data = {"answers": {}}
        if self.path and self.path.is_file():
            self.data = json.loads(self.path.read_text())
        self._lock = threading.Lock()
        self._added: set[str] = set()  # answers this process wrote

    def get(self, judge: str, question: Question, model: str | None = None):
        """The saved answer, or None. With a model, only an answer from that model counts."""
        entry = self.data["answers"].get(f"{judge}:{question.key()}")
        if entry is None or (model and entry.get("model") != model):
            return None
        return entry["answer"]

    def put(self, judge: str, model: str, question: Question, answer: object) -> None:
        with self._lock:
            self._added.add(f"{judge}:{question.key()}")
            self.data["answers"][f"{judge}:{question.key()}"] = {
                "question_id": question.id,
                "judge": judge,
                "model": model,
                "answer": answer,
            }

    def save(self, path: Path | None = None) -> Path:
        """Write the answers, merged with whatever another process saved to the
        same file meanwhile: only the answers this process added replace theirs."""
        import fcntl

        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target.with_name(target.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            on_disk = (
                json.loads(target.read_text()) if target.is_file() else {"answers": {}}
            )
            with self._lock:
                for key in self._added:
                    on_disk["answers"][key] = self.data["answers"][key]
                self.data = on_disk
                # Written: a later save must not put these back over a newer
                # answer another process saves for the same question.
                self._added.clear()
            written = target.with_name(target.name + ".tmp")
            written.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n")
            written.replace(target)
        return target


class JevJudge:
    """TypeSafe's Jev: typed answers with probabilities, no text generation."""

    name = "jev"

    def __init__(self, endpoint: Endpoint):
        if not endpoint.api_key:
            raise ValueError("Set ASSISTANTEVAL_READER_API_KEY to your TypeSafe key")
        self.api_key, self.model, self.identity = (
            endpoint.api_key,
            endpoint.model,
            endpoint.identity,
        )
        self.url = (endpoint.base_url or JEV_URL).rstrip("/") + "/systemone"

    def __call__(self, q: Question):
        body = {
            "type": "noul" if q.kind == "yes_no" else "choice",
            "instructions": q.text,
        }
        if q.criteria:
            body["criteria"] = q.criteria
        payload = json.dumps(
            {"model": self.model, "state": q.context, "questions": {"q": body}}
        ).encode()
        for attempt in range(JEV_TRIES):
            request = urllib.request.Request(
                self.url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    answer = json.load(response)["answers"]["q"]
                return answer["noul"] if q.kind == "yes_no" else answer["probabilities"]
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_CODES or attempt == JEV_TRIES - 1:
                    raise JudgeError(f"jev: HTTP {error.code}") from error
            # urlopen wraps only errors sending the request: a dropped
            # connection while reading, or a malformed body, arrives raw.
            except (
                OSError,
                http.client.HTTPException,
                ValueError,
                KeyError,
                TypeError,
            ) as error:
                if attempt == JEV_TRIES - 1:
                    # The detail tells a changed response schema from a dropped connection.
                    raise JudgeError(
                        f"jev: {type(error).__name__}: {str(error)[:200]}"
                    ) from error
            time.sleep(2**attempt)


_LLM_SYSTEM = """You answer one question about the STATE for a benchmark grader.
Answer only the literal question. Use only the STATE.
Return JSON only.
For a yes/no question: {"p_yes": <probability from 0 to 1 that the answer is yes>}
For a pick-one question: {"probabilities": {"<option>": <0 to 1>, ...}}"""


def _json_object(text: str) -> dict:
    """The JSON object in a reply, tolerating a code fence or words around it,
    which models without a JSON mode add."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start : end + 1])


class LLMJudge:
    """A model behind any OpenAI-compatible chat API, asked for probabilities as JSON."""

    name = "llm"

    def __init__(self, endpoint: Endpoint):
        self.endpoint, self.model, self.identity = (
            endpoint,
            endpoint.model,
            endpoint.identity,
        )

    def __call__(self, q: Question):
        from openai import OpenAIError

        user = f"STATE\n{json.dumps(q.context, indent=1)}\n\nQUESTION ({'yes/no' if q.kind == 'yes_no' else 'pick one'})\n{q.text}"
        if q.criteria:
            user += (
                "\n\nWHAT YES AND NO MEAN\n" if q.kind == "yes_no" else "\n\nOPTIONS\n"
            ) + json.dumps(q.criteria, indent=1)
        client = self.endpoint.client(max_retries=5)  # retries failed requests
        for attempt in range(MALFORMED_TRIES):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _LLM_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    extra_body=self.endpoint.extra_body,
                )
            except OpenAIError as error:
                raise JudgeError(f"llm: {error}") from error
            try:
                return _probabilities(q, response.choices[0].message.content or "")
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                if (
                    attempt == MALFORMED_TRIES - 1
                ):  # open models now and then drop the format
                    raise JudgeError(f"llm: {type(error).__name__}: {error}") from error


def _probabilities(q: Question, text: str):
    """The answer in a judge's reply: probabilities in [0, 1]. A pick-one
    answer may leave out its "probabilities" wrapper, but must name at least
    one option."""
    data = _json_object(text)
    if q.kind == "yes_no":
        answer = float(data["p_yes"])
        if not 0 <= answer <= 1:  # also NaN
            raise ValueError("p_yes is not a probability")
        return answer
    named = data.get("probabilities", data)
    if not set(named) & set(q.criteria):
        raise ValueError("the reply names no option")
    answer = {option: float(named.get(option, 0.0)) for option in q.criteria}
    # A little over 1 is rounding; more, or any value outside [0, 1], is not a distribution.
    if not all(0 <= p <= 1 for p in answer.values()) or sum(answer.values()) > 1.05:
        raise ValueError("the options' probabilities are not a distribution")
    return answer


def _identity(judge) -> str | None:
    """What a judge's saved answers are keyed by (`Endpoint.identity`), or its model."""
    return getattr(judge, "identity", None) or getattr(judge, "model", None)


def judge_for(endpoint: Endpoint | None):
    """The judge that asks the endpoint, or None for none."""
    if endpoint is None:
        return None
    return JevJudge(endpoint) if endpoint.api == "jev" else LLMJudge(endpoint)


def ladder_from_env(store: AnswerStore, offline: bool = False) -> Ladder:
    """The ladder the environment configures; offline, saved answers only."""
    if offline:
        return Ladder(store)
    return Ladder(
        store,
        reader=judge_for(endpoint("READER")),
        judge=judge_for(required("JUDGE")),
        checker=judge_for(endpoint("CHECKER")),
    )


@dataclass
class Ladder:
    """Decide each question at the lowest rung that is sure.

    A rung set to None is offline: only its saved answers are used.
    """

    store: AnswerStore
    reader: object | None = None
    judge: object | None = None
    # An independent model for re-checks that leave the judge out: when the
    # reader is unsure, it answers before a person does. Unset by default.
    checker: object | None = None
    calls: dict = field(default_factory=lambda: {"reader": 0, "judge": 0, "errors": 0})
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def models(self) -> dict:
        """Each rung's judge, for the verdict."""
        rungs = ("reader", "judge", "checker") if self.checker else ("reader", "judge")
        return {rung: _identity(getattr(self, rung)) for rung in rungs}

    def _answer(self, judge, rung: str, q: Question):
        """A saved answer from the same judge, else a new answer. None if there is neither."""
        model = _identity(judge)
        saved = self.store.get(rung, q, model)
        if saved is not None or judge is None:
            return saved
        try:
            answer = judge(q)
        except JudgeError:
            with self._lock:
                self.calls["errors"] += 1
            return None
        with self._lock:
            self.calls[rung] = self.calls.get(rung, 0) + 1
        self.store.put(rung, model or rung, q, answer)
        return answer

    def decide_one(self, q: Question) -> Decision:
        human = self.store.get(HUMAN, q)
        if human is not None:
            return Decision(q.id, human, HUMAN)
        answers = {}
        # A re-check that leaves the judge out goes to the checker, if set.
        checker = ("checker",) if self.checker and "judge" not in q.rungs else ()
        for rung in ("reader", "judge", *checker):
            if rung not in (*q.rungs, *checker):
                continue
            answers[rung] = answer = self._answer(getattr(self, rung), rung, q)
            if answer is not None:
                value, sure = read(q.kind, answer)
                if sure:
                    return Decision(q.id, value, rung, **answers)
        note = (
            "no saved answer"
            if not any(a is not None for a in answers.values())
            else "judges unsure"
        )
        return Decision(
            q.id, None, "needs_human_annotation", **answers, note=note, question=q
        )


class Asking:
    """ask(question): a Future of the question's decision on the ladder,
    decided in a pool. An identical question (the same key) is decided once."""

    def __init__(self, ladder: Ladder):
        self.ladder, self.pool = ladder, ThreadPoolExecutor(8)
        self.futures: dict[str, Future] = {}
        self.asked: dict[str, Question] = {}
        self._lock = threading.Lock()

    def __call__(self, q: Question) -> Future:
        with self._lock:
            self.asked.setdefault(q.id, q)
            if q.key() not in self.futures:
                self.futures[q.key()] = self.pool.submit(self.ladder.decide_one, q)
            return self.futures[q.key()]

    def __enter__(self) -> Asking:
        return self

    def __exit__(self, *exc) -> None:
        self.pool.shutdown()

    def decisions(self) -> list[Decision]:
        """One decision per question asked, named for it, in question order."""
        return [
            replace(
                d := self.futures[q.key()].result(),
                question_id=i,
                question=d.question and q,
            )
            for i, q in sorted(self.asked.items())
        ]
