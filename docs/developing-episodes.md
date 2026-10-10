# Developing tasks, assistants and services

This guide covers the task runner: how a task runs, and how to add an assistant or a service. Tasks themselves live in `tasks/` (see `tasks/README.md`); the run record the runner writes is specified in [run-record.md](run-record.md), and grading is separate (`assistanteval.grader`).

## The seams

| Component | Owns | Does not own |
|---|---|---|
| `assistanteval/run_task.py` | Step order, the clock and limits, the conversation loop, the run record | The assistant's internals, service internals, what the user says |
| An assistant adapter, `open_chat` in any module | Starting the assistant, its reset, connector setup, sending and reading messages, approval clicks | Service state, the task, the answer |
| The simulated user, `assistanteval/simulated_user.py` | What to say next, when to wait, when the conversation is over | The assistant or the services |
| A service, `assistanteval/services/<service>.py` | Its fixture, its session and key, every request it serves, its state | The assistant or the conversation |

They never call each other; the runner calls each in turn:

1. Open each of the task's services (`assistanteval.connectors.open_service`) and prepare it.
2. Open the adapter; `reset()` proves the assistant starts clean, `connect(connectors)` adds the task's connectors.
3. Converse: send the simulated user's opening, `read()` every second, and hand the user new messages once two reads agree and nothing new has appeared for a few seconds. An adapter that can tell its answer is still coming sets `answering` in `read()`, from the same snapshot as the messages, and the runner holds the user while it is set (the model baseline while its tool loop runs; the run deadline bounds a loop that never ends). A silent assistant is nudged; the run ends when the user ends it, or at the turn limit or deadline.
4. Close the adapter, collect each service's request log and final state, release the services, and save the run record.

Any failure ends the run as `setup_error`, which is never scored; `result.opening_sent` says whether a retry is safe. A failure of the simulated user's judge mid-run ends it as `judge_error`, also never scored.

## Run a task

```bash
export ASSISTANTEVAL_JUDGE_MODEL=... ASSISTANTEVAL_JUDGE_BASE_URL=... ASSISTANTEVAL_JUDGE_API_KEY=...
export ASSISTANTEVAL_BASELINE_MODEL=... ASSISTANTEVAL_BASELINE_BASE_URL=... ASSISTANTEVAL_BASELINE_API_KEY=...
uv run python -m assistanteval.run_task tasks/kingsway-reschedule-crew.json --assistant model_baseline
uv run python -m assistanteval.run_task tasks/kingsway-reschedule-crew.json --assistant my_package.adapter:open_chat
```

The models besides the assistant (the simulated user's reader, judge and writer) are set per role; see `assistanteval/endpoints.py`. The run's services start on a local HTTP server in the runner's process (see [Services](#services)).

## Add an assistant

1. Write a `<Name>Chat` class that implements the `Assistant` contract in `assistanteval/contracts.py`:
   - `reset() -> ResetEvidence`: put the assistant back to a clean start and prove it, from its own state where possible, never from its words. A harness started with fresh state for each run proves its reset by being new.
   - `connect(connectors) -> list[str]`: add the task's connectors; return any messages the assistant posted in the chat while doing so, so they stay out of the conversation. A connector carries its full `url` and the run's bearer `key`.
   - `send(text)`: deliver one user message, and raise if delivery cannot be confirmed: `EpisodeStepError("prompt_delivery_unconfirmed")` when the message may have arrived anyway, so the runner records it as sent and the run is not retried as fresh.
   - `read() -> (messages, clicks)`: every assistant message since the opening, oldest first, each with an id that stays the same when the assistant edits the message, and each approval prompt this read answered. Never wait or sleep in `read()`; the runner owns timing.
2. Add an `open_chat(account, evidence_dir)` context manager that starts the assistant and yields the chat. The runner passes `account=None`. Set `open_chat.connector_kind` (`openapi_api_key` or `mcp_bearer`), and optionally `open_chat.manifest`, a function returning what the run measured (model, harness version and configuration hash), which the runner adds to the record's manifest. Clean up the run's connectors when the context closes.
3. Run it as `--assistant <module>:open_chat`; an adapter can live in its own package. A built-in adapter goes in `assistanteval/assistants/<name>.py` and `run_task.ASSISTANTS`.
4. Test it against a fake transport. `tests/test_architecture_boundaries.py` checks that built-in adapters import no service code.

## Built-in assistants

The model baseline, `ModelBaselineChat` in `assistanteval/assistants/model_baseline.py` (`--assistant model_baseline`), measures a model on its own, with no product around it: any model behind an OpenAI-compatible chat API, set as the BASELINE role (`ASSISTANTEVAL_BASELINE_MODEL`, `_BASE_URL`, `_API_KEY`), in a minimal tool-calling loop. Each connector becomes tools generated from the service's `/openapi.json`; MCP connectors are not supported yet. The system prompt is short and fixed and gives only the user's local time; the manifest records the model, its base URL, the harness version and the prompt hash, since those are what a run measures. Each run starts a new chat, which is its reset. The loop runs on a worker thread, so `read` never waits; each text the model writes is one message. A model that keeps calling tools for 40 steps goes quiet and is nudged like a stalled assistant, while a model API failure is a `setup_error`. The loop's full working (every message, tool result and token count) is saved as `evidence/model_baseline_trace.json`. Permission clicks are always 0, since nothing asks it to approve anything.

The reference assistant, `ReferenceChat` in `assistanteval/assistants/reference.py` (`--assistant reference`), is a fixed script per task. It shows what a passing run looks like and lets the whole pipeline run with no model or API key. It uses the task's services through their REST API with the connector it is given, as an assistant would, and answers each message at once. It recognises its tasks by their opening message (see the module's docstring). Each script is a generator that yields the assistant's answers and receives the user's replies. It sends only after a clear yes to the draft the user last saw. A no that carries a correction ("no, tell her the rain cover isn't included") gets a revised draft, a plain no gets "what should I change?", and an unclear reply is asked again.

## Services

`assistanteval/connectors.py` is what the runner calls to open a service. Every service runs on one local HTTP server, started in the runner's process on first use: service `s` serves REST at `<base>/s` and MCP (streamable HTTP) at `<base>/s/mcp`, each behind the run's bearer key.

| Variable | Default | |
|---|---|---|
| `ASSISTANTEVAL_SERVICES_HOST` | `127.0.0.1` | Where the server listens. Use `0.0.0.0` for an assistant in a container. |
| `ASSISTANTEVAL_SERVICES_PORT` | a free port | |
| `ASSISTANTEVAL_SERVICES_URL` | `http://<host>:<port>` | The base URL the assistant is given, when it reaches this machine by another name, such as `http://host.docker.internal:8765`. |
| `ASSISTANTEVAL_SERVICES_STORE` | a new temporary directory | The session directory. |

A caller with services elsewhere passes its own `services_from` to `run_task.run_task()`, returning objects with `ServiceSession`'s methods.

`assistanteval/world.py` keeps one session per service per run: a JSON object with the service, a hash of its bearer key, the start and current state, the request log, and whether it is revoked. Sessions live in a local directory (`DirBackend`, or any backend with the same `get` and `put`), and every call validates, applies and logs in one compare-and-swap write, so the log is complete and a refused call changes nothing. The public APIs expose service-shaped actions, never seed, reset, snapshot or log routes. `ServiceSession` prepares the session, reads its state and log, and revokes the key and proves it no longer opens the session. Grading reads the start and end state and the log from the run record, not from the live service.

Current fixtures are synthetic. Service handlers do not add bearer keys or auth headers to action payloads; each service redacts its own sensitive values in the request log. Other agent-supplied domain text remains visible; inspect a run record before sharing it.

## Add a service

1. Write `assistanteval/services/<service>.py`: a pydantic state model (it validates the fixture, so a bad fixture fails before the episode starts), `service(name, title, State)`, and one function per operation, declared with `@<service>.op(name, "METHOD /path", tool=, params=, output=, kind=, plain=, description=)`. The function takes the state and the validated params, changes the state in place and returns the output; it raises `Refused(message, status)` for a refusal the assistant should see. Anything else it raises becomes `503 <Title> fixture unavailable`, since validation errors can contain private fixture values. The fixture a session starts from is the task's fixture rendered at the run's anchor (see [tasks/README.md](../tasks/README.md#dates)), so its dates are current and the service never sees a `{{placeholder}}`. `kind` is the grader's approval class, and `plain` names the change in the user's words (every change has them, a read has none); every logged request carries them into the run record. The helpers `find`, `new_id` and `matches` in `assistanteval/services/__init__.py` cover lookups, new ids and searches.
   - **REST** comes from the declaration: path fields from the `{placeholders}`, other inputs as query parameters for GET and the JSON body otherwise, with the typed schema in `openapi.json`.
   - **MCP** too: one tool per operation, with the same inputs and read-only or destructive annotations.
   - Both transports go through `world.invoke`, so a call is logged the same way either way and a refusal shows the assistant the same message. A revoked or expired key is `Invalid API key` (401) on both.
2. Import the module in `assistanteval/services/__init__.py`, so a task's `environment.connectors` can name the service. Add a task fixture for it, and test its own rules in `tests/test_service_rules.py`. The service should not require runner or assistant changes. `tests/parity/` pins each service's behaviour call by call. Goldens for a new service are recorded from the service itself with `uv run python -m tests.parity.record_new <service>`: add its script to `tests/parity/scripts.py` and the service to `record_new.NEW_SERVICES`.

Email fixtures can declare `allowed_senders` and `allowed_recipients`; `POST /v1/messages` accepts exactly one `sender`, `recipient`, `subject`, and `body` from those lists and appends a synthetic message to that session's `sent` array. It never delivers mail. `GET /v1/messages?query=...` searches inbox and sent messages and returns the allowed addresses.

## Verify changes locally

```bash
uv sync
uv run pytest -q
```

### The end-to-end test

`tests/test_end_to_end.py` runs the whole pipeline offline, in about a second, as part of `uv run pytest -q`: task file, services, reference assistant, simulated user, run record, grader verdict. For each reference task it:

1. Serves the task's services in process (`tests/conftest.in_process_services`), so the assistant's requests reach them without a network.
2. Runs `run_task` with the reference assistant and the real `SimulatedUser` (its rules, fact reveals and approvals), and saves the run record.
3. Loads the record back and grades it with the real grader against the task file.

Only the judges are stand-ins: `KeywordJudge` reads text by keyword for both the simulated user and the grader, written to read these transcripts as a careful reader would. The clock is pinned to set anchors, for the runner and the service engine alike, so the test does not depend on today's date. The tasks run at different anchors (a Tuesday morning, a Wednesday night and a Saturday afternoon), so worlds are checked on different days of the week.

`tests/test_amazon_end_to_end.py` runs `battery-size-check`, `tile-order-card-declined` and `tile-order-card-backup` the same way on the local `amazon` service; careless variants, such as one that orders at once, fail as they should.

It also runs careless scripts, defined in the test, that make the mistakes observed in real assistants: they act without asking, assume the rain cover, take an offer below the floor, drop Luis from Kingsway, and claim work never done. A good run passes every validity check and gate, and each careless run fails on exactly the gates it should. It does not prove that real judges answer these questions the same way; that is what calibration and human annotation are for.

`tests/test_services.py` also starts the real local server and connects to a service with the official MCP client over streamable HTTP, as an agent harness would.
