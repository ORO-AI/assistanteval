"""One Harbor trial of an exported task, end to end, with no model keys.

    uv run python -m tests.harbor_smoke.smoke      # needs Docker and network access

The target is a scripted ACP agent (`scripted_acp.py`) behind Harbor's real ACP
bridge and acpx; the simulated user is AssistantEval's, with keyword judges
standing in for models; the services and verifier containers are the exported
ones. It checks that the user agent drove the conversation, that the target's
MCP call reached the services container and the run record, and that the
verifier graded the record (offline, with no JUDGE role set).
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.protocols.acp import ACPAgentMixin
from harbor.models.bridge import BridgeConfig, BridgeKind
from harbor.models.trial.config import (
    AgentConfig,
    TaskConfig,
    TrialConfig,
    UserAgentConfig,
)
from harbor.trial.trial import Trial

from assistanteval import run_task
from assistanteval.harbor import export
from assistanteval.simulated_user import SimulatedUser
from tests.test_end_to_end import KeywordJudge

SCRIPT = Path(__file__).with_name("scripted_acp.py")
TASK = run_task.ROOT / "tasks/kingsway-reschedule-crew.json"


class KingswayJudge(KeywordJudge):
    def meets(self, turn: str, condition: str) -> bool:
        return (
            "kingsway" in turn.lower()
        )  # the rule: change only Kingsway or Fraserview


class ScriptedTarget(BaseAgent, ACPAgentMixin):
    capabilities = AgentCapabilities(
        bridges=frozenset({BridgeKind.ACP}), mcp_servers=True
    )

    @staticmethod
    def name() -> str:
        return "scripted-acp"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        await environment.upload_file(SCRIPT, "/opt/scripted_acp.py")

    async def run(self, instruction, environment, context) -> None:
        raise NotImplementedError("only an ACP target")

    def acp_command(self) -> list[str]:
        return ["python3", "/opt/scripted_acp.py"]

    def acp_env(self) -> dict[str, str]:
        return {"ASSISTANTEVAL_SMOKE": "bridged"}  # how a harness gets its keys


async def trial(task_dir: Path, trials: Path):
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials,
        agent=AgentConfig(import_path="tests.harbor_smoke.smoke:ScriptedTarget"),
        user_agent=UserAgentConfig(
            import_path="assistanteval.harbor.user:SimulatedUserAgent",
            bridge=BridgeConfig(kind="acp"),
        ),
    )
    return await (await Trial.create(config)).run()


def main() -> int:
    run_task.simulated_user = lambda task: SimulatedUser(task, KingswayJudge(task))
    scratch = Path(tempfile.mkdtemp(prefix="assistanteval-harbor-smoke-"))
    print("trial files:", scratch)
    task_dir = export.export(TASK, scratch / "tasks", export.build_wheel(scratch))
    result = asyncio.run(trial(task_dir, scratch / "trials"))
    print(
        result.model_dump_json(indent=1, include={"exception_info", "verifier_result"})
    )
    assert result.exception_info is None, result.exception_info
    [run_json] = scratch.glob("trials/*/user-agent/*/run.json")
    record = json.loads(run_json.read_text())
    assert record["run_id"] == run_json.parents[2].name  # the trial, unique in a job
    turns = record["turns"]
    operations = [r["operation"] for s in record["services"] for r in s["requests"]]
    print(
        "end:",
        record["result"]["end_reason"],
        "turns:",
        len(turns),
        "requests:",
        operations,
    )
    assert [t["role"] for t in turns[:2]] == ["user", "assistant"]
    assert "calendar.search" in operations
    assert "(bridge env: bridged)" in turns[1]["text"]
    assert result.verifier_result.rewards["needs_human_annotation"] == 1  # no judge set
    print("Harbor smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
