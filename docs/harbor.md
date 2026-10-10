# Running AssistantEval in Harbor

[Harbor](https://github.com/harbor-framework/harbor) runs a task against any agent harness, locally in Docker or on cloud sandboxes, and its task format feeds several RL trainers. AssistantEval tasks run there as simulated-user trials: AssistantEval's simulated user plays the user, the harness under test is Harbor's ACP target, and AssistantEval's grader is the verifier. Pinned to Harbor 0.24.0: from a checkout, `uv sync --extra harbor`; elsewhere, `pip install 'assistanteval[harbor]'`.

## Run a task

```bash
uv sync --extra harbor
uv run assistanteval harbor-export tasks/ harbor-tasks/

# The simulated user runs on the Harbor host and reads its model roles there.
export ASSISTANTEVAL_JUDGE_MODEL=... ASSISTANTEVAL_JUDGE_BASE_URL=... ASSISTANTEVAL_JUDGE_API_KEY=...
export ASSISTANTEVAL_READER_MODEL=... ASSISTANTEVAL_WRITER_MODEL=...   # optional, as in a local run

# A harness Harbor already drives over ACP: claude-code, codex, gemini-cli or opencode
uv run harbor run -p harbor-tasks/kingsway-reschedule-crew \
    --agent opencode --model openrouter/<provider>/<model> \
    --user-agent assistanteval.harbor.user:SimulatedUserAgent --bridge acp --yes

# Hermes Agent, through AssistantEval's target class
uv run harbor run -p harbor-tasks/kingsway-reschedule-crew \
    --agent assistanteval.harbor.targets:HermesTarget --model <provider>/<model> \
    --user-agent assistanteval.harbor.user:SimulatedUserAgent --bridge acp --yes
```

The harness's own credentials follow Harbor's rules for that agent (for Hermes: a native provider key such as `ANTHROPIC_API_KEY` with `anthropic/<model>`, else `OPENROUTER_API_KEY` with an OpenRouter model id). `--yes` confirms passing the JUDGE and READER settings to the verifier, which grades in its own container.

## What a trial does

| Part | Where | What |
|---|---|---|
| `instruction.md` | Harbor host, to the user agent only | The task and its fixtures. In a simulated-user trial the target never sees the instruction. |
| `environment/` `main` | Container | The harness, installed by Harbor, with the task's time zone (`TZ`). |
| `environment/` `services` | Container on the task's network | `assistanteval serve`: every service over REST and MCP, with an admin API. |
| `task.toml` `[[environment.mcp_servers]]` | | `http://services:8000/<service>/mcp` for each task service; Harbor registers them with the harness. |
| `SimulatedUserAgent` | Harbor host | Runs the task with `run_task.run_task`, as a local run does. It claims the services' admin token, opens a session per service, talks to the harness through the bridge's `acpx` (`AcpChat`), collects each service's state and requests, and uploads the run record to `/logs/artifacts/assistanteval/run`. |
| `tests/` | Separate verifier container | `assistanteval harbor-verify` grades the record and writes `verdict.json` and `reward.json`. |

The services' admin API (`assistanteval/remote.py`) is guarded by a token the user agent claims before the harness receives its first message; the first claim wins, so the harness cannot open, read, reset or revoke sessions. Harbor's task MCP servers have no header field, so a service request without a bearer key uses that service's latest session; the trial's private network isolates the run.

## Reading rewards

`reward.json` has `reward` (1 for a pass, else 0) and a 0/1 metric per outcome: `pass`, `fail`, `fail_unsafe`, `needs_human_annotation` and `invalid`. Leave `invalid` and `needs_human_annotation` trials out of a pass rate, as `assistanteval report` does. With no JUDGE set for the verifier, it grades with code checks and saved answers only, and the rest waits for annotation (`assistanteval annotate` on the trial's `verifier/verdict.json`).

## Tests

- `tests/test_harbor.py`, `tests/test_acp.py`, `tests/test_remote.py`: the parts, in the unit suite.
- `tests/harbor_smoke/smoke.py`: one real Harbor trial in Docker with no model keys. The target is a scripted ACP agent behind Harbor's real bridge and acpx. It checks that the simulated user drives the conversation, that the target's MCP call reaches the services container and the record, that the bridge's environment (how a harness gets its keys) reaches the target, and that the verifier grades the record. `uv run python -m tests.harbor_smoke.smoke` (about a minute once images are cached).

## Hermes Agent

`HermesTarget` (`assistanteval/harbor/targets.py`) installs Hermes as Harbor 0.24 does and runs `hermes acp` with Harbor's configuration for it, with these differences:

- It checks the install with `hermes --version`; Harbor's check runs `hermes version`, which current Hermes lacks.
- It installs into `HERMES_HOME=/tmp/hermes`, the home Hermes runs with. Hermes keeps one dependency environment per home, and one built later for ACP lacks the MCP SDK, so Hermes would skip every MCP server.
- It waits up to 30s (`mcp_discovery_timeout`, 1.5s by default) for MCP servers when Hermes builds a session's tools. The services list their MCP tools before their sessions open, for a harness that starts before the run.
- It starts Hermes behind `acp_resume_shim.py`. Hermes does not keep an empty ACP session, and resuming an id it does not know opens a session under another id, so it would refuse every prompt acpx resumes. The shim opens a session for that resume and translates between the two ids.

Hermes's stderr is saved as `agent/hermes-acp.log` in each trial.

## Limitations

- OpenClaw (`openclaw acp`) and Goose (`goose acp`) need target classes like `HermesTarget`.
- Turn-taking is ACP's: a turn ends when the harness's prompt returns. The runner's settle, nudge and deadline rules apply as in a local run.
- The runner sets the process's `TZ` for the task. AssistantEval's own date handling uses explicit time zones, but with concurrent trials the rest of the Harbor process (log timestamps, say) sees whichever task set it last.
- Task dates render when the user agent starts the run, so a task stays current; replays for RL need a fixed anchor.
