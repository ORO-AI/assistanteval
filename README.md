# AssistantEval

A verifiable benchmark for personal AI assistants. Results, methodology and every published run: [assistanteval.com](https://assistanteval.com).

Each task puts an assistant in a small synthetic world: a calendar, a mailbox, a marketplace and so on, served to it over REST or MCP. A simulated user talks to it over several turns, gives private facts only when asked, and approves only what its rules allow. Every service records what the assistant did, and the grader scores the run from that record alone.

The assistant can be any model in a minimal tool loop, an open agent harness you adapt, or your own product. The models that play the user's reader and writer and that judge the run are yours to choose too: any OpenAI-compatible API, hosted or local.

## How a run works

1. **Task** (`tasks/<id>.json`): the world's fixtures, the opening message, what only the user knows, when the user approves an action, and the grading rules. See [tasks/README.md](tasks/README.md).
2. **Runner** (`assistanteval/run_task.py`): starts the task's services on a local server with fresh per-run keys, resets the assistant, connects the services, and runs the conversation with the simulated user (`assistanteval/simulated_user.py`).
3. **Run record** (`runs/<run_id>/`): the conversation, every service request with its input and output, the start and end state, the simulated user's decisions, and which models played each role. See [docs/run-record.md](docs/run-record.md).
4. **Grader** (`assistanteval/grader/`): validity checks, then rules answered by code or by a judge ladder (a saved human answer, the reader, the judge, then a human annotation queue), then a pass-rate report. See [docs/grader/README.md](docs/grader/README.md).

## Run it

```bash
uv sync
uv run pytest -q          # includes an offline end-to-end run: no API key needed

# Set the judges and the simulated user's models: see "Two setups" below.
# The assistant: here a model in the built-in tool loop
export ASSISTANTEVAL_BASELINE_MODEL=<model> ASSISTANTEVAL_BASELINE_BASE_URL=<url> ASSISTANTEVAL_BASELINE_API_KEY=<key>

uv run assistanteval run tasks/kingsway-reschedule-crew.json --assistant model_baseline
uv run assistanteval grade runs/<run_id>
uv run assistanteval report runs/ --out grades/
```

To run tasks in [Harbor](https://github.com/harbor-framework/harbor) against an agent harness such as Hermes Agent or OpenCode, see [docs/harbor.md](docs/harbor.md).

`--assistant` also takes `module:factory`, so an adapter for your own harness or product can live in its own package. See [docs/developing-episodes.md](docs/developing-episodes.md) for the assistant contract, the services and how to add a task or a service.

## Model roles

| Role | Does | Required |
|---|---|---|
| `JUDGE` | Answers what the reader is unsure about, and checks proposals against the user's approval rules | For a live run or live grading |
| `READER` | Reads each assistant turn for the simulated user, and answers the grader's questions first | No; without it the judge reads too |
| `WRITER` | Rewords the simulated user's fixed replies in the persona's voice | No; without it the fixed replies are sent |
| `BASELINE` | The model under test in the built-in tool loop | For `--assistant model_baseline` |

Each role reads `ASSISTANTEVAL_<ROLE>_MODEL`, `_BASE_URL`, `_API_KEY`, and optionally `_API` (`openai`, or `jev` for TypeSafe's Jev as the reader) and `_EXTRA_BODY`. Records keep each role's model and base URL, never its key. Scores from different judge models are not directly comparable: report the judges with the scores.

### Two setups

**Official**: the judges of the published AssistantEval results. Use it to compare with them.

```bash
export ASSISTANTEVAL_READER_API=jev ASSISTANTEVAL_READER_MODEL=jev-1.13.0 ASSISTANTEVAL_READER_API_KEY=<TypeSafe key>
export ASSISTANTEVAL_JUDGE_MODEL=gpt-5.6-sol ASSISTANTEVAL_JUDGE_API_KEY=<OpenAI key>
export ASSISTANTEVAL_WRITER_MODEL=z-ai/glm-5.3-flash ASSISTANTEVAL_WRITER_BASE_URL=https://openrouter.ai/api/v1 \
       ASSISTANTEVAL_WRITER_API_KEY=<OpenRouter key> ASSISTANTEVAL_WRITER_EXTRA_BODY='{"reasoning": {"effort": "low"}}'
```

**OpenRouter**: one key, for development. The same judge and writer models; the reader is TypeSafe's Jev Router, which picks its model for each request rather than pinning `jev-1.13.0`. Its readings can differ from `jev-1.13.0`'s, so compare with published results only under the official setup.

```bash
OR=https://openrouter.ai/api/v1 KEY=<OpenRouter key>
export ASSISTANTEVAL_READER_MODEL=typesafe/jev-router ASSISTANTEVAL_READER_BASE_URL=$OR ASSISTANTEVAL_READER_API_KEY=$KEY
export ASSISTANTEVAL_JUDGE_MODEL=openai/gpt-5.6-sol ASSISTANTEVAL_JUDGE_BASE_URL=$OR ASSISTANTEVAL_JUDGE_API_KEY=$KEY
export ASSISTANTEVAL_WRITER_MODEL=z-ai/glm-5.3-flash ASSISTANTEVAL_WRITER_BASE_URL=$OR ASSISTANTEVAL_WRITER_API_KEY=$KEY \
       ASSISTANTEVAL_WRITER_EXTRA_BODY='{"reasoning": {"effort": "low"}}'
```

To choose other judges, see [docs/grader/README.md](docs/grader/README.md#choosing-the-judges).

## Repository layout

| Path | What it is |
|---|---|
| `tasks/` | The task catalog and its fixtures. |
| `assistanteval/run_task.py`, `contracts.py`, `run_record.py`, `task.py`, `dates.py` | The runner, the assistant contract, the run record, the task model, and the renderer of a task's dates at a run's start. |
| `assistanteval/endpoints.py` | The model roles. |
| `assistanteval/simulated_user.py` | The simulated user. |
| `assistanteval/grader/` | Grading, annotation and the pass-rate report. |
| `assistanteval/services/`, `world.py`, `connectors.py` | The synthetic services, the engine that serves them over REST and MCP, and the local server the runner opens them on. |
| `assistanteval/assistants/` | Built-in assistants: the model baseline and a scripted reference assistant. |
| `assistanteval/harbor/`, `remote.py`, `assistants/acp.py` | Running tasks in Harbor against any ACP harness. See [docs/harbor.md](docs/harbor.md). |
| `docs/frameworks.md` | Which open eval and RL frameworks to integrate with, and why. |

## Safety rails

- Assistants only ever reach synthetic services, with per-run keys that are revoked, and proven revoked, after every run. In Harbor, where a task's MCP servers carry no key, the trial's private network isolates the run instead.
- No service delivers mail; the `email` service also sends only to its fixture's allowed recipients.

## License

The code is licensed under the [Apache License 2.0](LICENSE). The tasks and their fixtures in `tasks/` are licensed under [CC BY 4.0](tasks/LICENSE): you may share and adapt them, including commercially, with attribution to AssistantEval by ORO AI. See [NOTICE](NOTICE).
