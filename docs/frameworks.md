# Open eval and RL frameworks: which to integrate with

As of 2026-10-09. Harbor is the first integration ([harbor.md](harbor.md)); this page records the comparison behind that choice.

## What AssistantEval needs from a framework

A run here differs from most agent benchmarks in four ways:

1. **A simulated user over many turns.** The assistant sees only the opening message and the user's replies. The user is rule-bound: it gives a private fact only when asked, approves only actions its rules allow, and nudges when the assistant goes quiet.
2. **Several stateful services per task** (calendar, email, marketplace and others), each served over REST and MCP behind a per-run bearer key, with every request logged with its result.
3. **A black-box assistant.** It can be a bare model in our tool loop, or a full agent harness (Hermes Agent, OpenClaw, Goose, nanobot) in its own container that does its own tool calls.
4. **Grading from the recorded run alone**: code checks over the service log, then judge questions answered with probabilities, then human annotation. Outcomes: pass, fail, fail_unsafe, needs_human_annotation, invalid.

## Comparison

| | Harbor | verifiers (Prime Intellect) | OpenEnv | NeMo Gym | Inspect AI | tau2-bench |
|---|---|---|---|---|---|---|
| License, latest | Apache-2.0, v0.24.0 (2026-10-05) | MIT, v0.3.1; `main` is a v1 rewrite | BSD-3, v0.8.0 | Apache-2.0, v0.7.0 | MIT | MIT, v1.0.1 |
| Several services | Docker Compose sidecars; the verifier can collect files from them | MCP toolsets or external URLs | One container per environment | Resource servers | Compose sandboxes | In-process Python |
| Simulated user | Since 0.24: a second agent drives the target over ACP (`UserAgentConfig`) | `UserSimEnv`, or a custom `Env.run()` | No | Yes (tau2 servers) | No built-in | Yes, the reference design |
| Harness as the agent | Built-in agents include Hermes, OpenClaw, Goose and a generic ACP agent | Harnesses over ACP (Hermes, OpenClaw, Claude Code) | Through its Harbor wrapper | Several | CLI agents via a bridge | No |
| Grading | Any script in `tests/`; RewardKit LLM judge | `@vf.reward`, `vf.Judge` | Rubrics | `verify()` per server | Model-graded scorers | DB state and assertions |
| RL path | SkyRL, Tinker, verifiers, OpenEnv, NeMo Gym, rLLM consume Harbor tasks | prime-rl, with token capture | TRL, Unsloth, SkyRL, ART | NeMo RL, veRL | Eval only | Gymnasium wrapper |

Sources: [Harbor](https://github.com/harbor-framework/harbor) (moved from laude-institute), [verifiers](https://github.com/PrimeIntellect-ai/verifiers), [OpenEnv](https://github.com/huggingface/OpenEnv), [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), [Inspect](https://inspect.aisi.org.uk), [tau2-bench](https://github.com/sierra-research/tau2-bench), [ACP](https://agentclientprotocol.com).

Checked against source on 2026-10-09: Harbor's release dates, `UserAgentConfig` (`src/harbor/models/trial/config.py`), `MCPServerConfig` having no header or auth field (`src/harbor/models/task/config.py`), its installed `hermes`, `openclaw`, `goose` and `acp` agents, and verifiers' `v1/envs/user_sim` and `v1/tasksets/harbor`. The rest is from the projects' docs and is unverified.

## Fit, framework by framework

**Harbor.** Harbor's task directory is becoming the interchange format: several RL trainers consume it. Since 0.24 a custom user agent can drive the target over ACP, which carries the simulated user without changing the task. [harbor.md](harbor.md) describes the integration. What it had to work around:

- **Multi-turn harnesses.** Only harnesses that declare the ACP bridge take more than one instruction. Hermes, OpenClaw and Goose are installed agents that take one, so each needs a target class (`HermesTarget` exists).
- **Per-run keys.** MCP servers in a task config carry no auth header, so a keyless request uses the service's latest session, and the task's private network isolates the run.
- **No user behind a tool.** Harbor's tau3-bench adapter exposes the user as an MCP tool and explains it in the instruction. That breaks our rule that the assistant sees only the user's messages, so the user is the ACP user agent instead.
- **Churn.** 0.24 broke verifier interfaces; the integration pins 0.24.0.

**verifiers.** The most direct RL path with real harnesses. `Env.run(task, agents)` lets us run our own conversation loop, Hermes and OpenClaw run as live ACP sessions, and verifiers captures tokens for prime-rl. Its Harbor taskset takes only single-instruction tasks, so a Harbor export alone does not carry the simulated user here. `main` is pre-release.

**OpenEnv.** A `reset`/`step` environment over WebSocket: tools are service operations, and the user's reply is the observation after an assistant message. Cheap, and it opens TRL and ART, but it covers bare models only. Harnesses reach it through its Harbor wrapper.

**NeMo Gym.** Capable, but Python 3.13.14+, Ray and a Responses-API focus make it a second heavy stack.

**Inspect AI.** Strong logs and sandboxes, no user simulator, no RL path.

**tau2-bench.** The design closest to ours (user simulator, tools, grading from database state, pass^k). Others adopted it by porting it, not by depending on it. A model to mirror, not a dependency.

## Recommendation

1. **Harbor**, as the distribution format and the harness × model evaluation path. Integrated: see [harbor.md](harbor.md).
2. **verifiers**, as the RL path: a taskset of the tasks and an `Env` that runs the simulated user's loop, with `grade()` as the reward.
3. **AssistantEval's own runner** for commercial assistant products and wall-clock behaviour that no framework models (edited messages, nudges after silence).
4. **ACP** as the protocol for driving harnesses, in AssistantEval's runner too, so one harness adapter serves all three paths.

## Open points for an integration

- **Turn-taking.** Under ACP a turn ends when the harness's prompt returns; see the Harbor [limitations](harbor.md#limitations).
- **A trainer's own loop.** RL frameworks that own the rollout loop need a step view of a run (`reset(task, anchor)`, `step(assistant_message) -> (user_reply, done)`), which AssistantEval does not offer yet.
- **Cost.** Every rollout also calls the simulated user's and the grader's models.
- **Held-out tasks.** A task published to Harbor's registry exposes its `tests/`, which hold the grading rules.
