"""Open harnesses as the ACP target of a simulated-user trial.

Harbor's ACP bridge already drives claude-code, codex, gemini-cli and opencode
as targets. A class here adds a harness Harbor installs but cannot yet target:
Harbor's own adapter installs it, and ``acp_install`` writes the configuration
that adapter's ``run`` would (model, provider, the task's MCP servers) without
running it, so acpx can start the harness as an ACP server.

    harbor run -p <task> --agent assistanteval.harbor.targets:HermesTarget \\
        --model <provider>/<model> \\
        --user-agent assistanteval.harbor.user:SimulatedUserAgent --bridge acp
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import yaml
from harbor.agents.installed.hermes import _NATIVE_PROVIDERS, Hermes
from harbor.agents.protocols.acp import ACPAgentMixin
from harbor.environments.base import BaseEnvironment
from harbor.models.bridge import BridgeKind

HERMES_HOME = "/tmp/hermes"  # where Harbor's Hermes adapter keeps its state
SHIM = Path(__file__).with_name("acp_resume_shim.py")
SHIM_PATH = "/opt/assistanteval/acp_resume_shim.py"
HERMES_PATH = 'export PATH="$HOME/.local/bin:$PATH"'


class HermesTarget(Hermes, ACPAgentMixin):
    """Hermes Agent over ``hermes acp``, configured as Harbor's ``hermes`` agent
    configures ``hermes chat``. A native provider key (as Harbor's adapter
    lists them) is used when set; otherwise the model name is an OpenRouter
    id, with ``OPENROUTER_API_KEY``."""

    capabilities = Hermes.capabilities.model_copy(
        update={"bridges": frozenset({BridgeKind.ACP})}
    )

    def get_version_command(self) -> str | None:
        return f"{HERMES_PATH}; hermes --version"

    async def install(self, environment: BaseEnvironment) -> None:
        """Harbor 0.24's install, ending with ``hermes --version`` (current
        Hermes has no ``hermes version`` command, so Harbor's fails), and run
        with the ``HERMES_HOME`` Hermes later runs with: Hermes keeps one
        dependency environment per home, and one made later lacks the MCP SDK,
        so Hermes silently skips every MCP server."""
        await self.ensure_system_dependencies(
            environment, ("curl", "git", "ripgrep", "xz")
        )
        branch = f" --branch {self._version}" if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; curl -fsSL https://raw.githubusercontent.com/"
                "NousResearch/hermes-agent/main/scripts/install.sh"
                f" | bash -s -- --skip-setup{branch} && {HERMES_PATH} && "
                f"mkdir -p {HERMES_HOME}/sessions {HERMES_HOME}/skills {HERMES_HOME}/memories && "
                "hermes --version"
            ),
            env={"HERMES_HOME": HERMES_HOME},
        )

    def _provider(self) -> tuple[dict[str, str], str, str]:
        """The harness's environment, model and provider for the model name."""
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        provider, model = self.model_name.split("/", 1)
        env = {"HERMES_HOME": HERMES_HOME, "TERMINAL_ENV": "local"}
        flag, keys = _NATIVE_PROVIDERS.get(provider, (None, []))
        key = next((k for k in keys if os.environ.get(k)), None)
        if key and flag:
            env[key] = os.environ[key]
            if provider == "openai" and os.environ.get("OPENAI_BASE_URL"):
                env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
            return env, model, flag
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("Set OPENROUTER_API_KEY, or a native provider's key")
        env["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        return env, self.model_name.removeprefix("openrouter/"), "openrouter"

    def acp_env(self) -> dict[str, str]:
        return self._provider()[0]

    def acp_command(self) -> list[str]:
        # Through the shim: Hermes forgets the empty session acpx resumes. Its
        # stderr goes to the agent logs Harbor keeps.
        log = f"ACP_SHIM_STDERR={self.environment_logs_dir.as_posix()}/hermes-acp.log"
        return [
            "env",
            log,
            "python3",
            SHIM_PATH,
            "bash",
            "-c",
            f"{HERMES_PATH} && exec hermes acp",
        ]

    async def acp_install(self, environment: BaseEnvironment) -> None:
        env, model, provider = self._provider()
        config = yaml.safe_load(self._build_config_yaml(model, self.options.max_turns))
        config["provider"] = provider
        # Hermes freezes a session's tools when it builds it, waiting only this
        # long (default 1.5s) for MCP servers: the task's services are its tools.
        config["mcp_discovery_timeout"] = 30
        text = yaml.dump(config, default_flow_style=False)
        commands = [
            f"mkdir -p {HERMES_HOME} && printf '%s' {shlex.quote(text)} > {HERMES_HOME}/config.yaml",
            self._build_register_mcp_servers_command(),
            # Hermes installs its ACP support on request.
            f"{HERMES_PATH} && cd {HERMES_HOME}/hermes-agent && "
            "python -c \"import pm; pm.sync_venv(['acp'], explicit=True)\"",
        ]
        for command in filter(None, commands):
            await self.exec_as_agent(environment, command=command, env=env)
        await environment.exec(f"mkdir -p {Path(SHIM_PATH).parent}", user="root")
        await environment.upload_file(SHIM, SHIM_PATH)
