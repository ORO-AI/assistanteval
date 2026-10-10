"""Guard the Python import boundaries between the runner, adapters and services."""

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = ".".join(path.relative_to(ROOT).parts[: -node.level])
                module = f"{package}.{module}".rstrip(".")
            if node.module and module != "assistanteval":
                names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def _foreign(imports: set[str], allowed: tuple[str, ...]) -> set[str]:
    return {
        module
        for module in imports
        if module == "assistanteval"
        or module.startswith("assistanteval.")
        and not any(module == name or module.startswith(name + ".") for name in allowed)
    }


def test_runner_imports_no_product_cloud_or_model_code():
    """The runner reaches products and services only through factories named
    in its tables; model SDKs stay inside the simulated user and judges."""
    path = ROOT / "assistanteval" / "run_task.py"
    blocked = {
        module
        for module in _imports(path)
        if module.startswith(("tools.", "boto3", "playwright", "openai", "anthropic"))
    }
    assert not blocked, f"assistanteval/run_task.py imports {blocked}"


@pytest.mark.parametrize(
    "module, banned",
    [
        # Records' codes load without product, cloud or runner code.
        ("assistanteval.contracts", {"playwright", "boto3", "assistanteval.run_task"}),
        # The engine needs only pydantic.
        ("assistanteval.world", {"fastapi", "starlette", "mcp", "mangum", "boto3"}),
    ],
)
def test_module_loads_without(module, banned):
    probe = f"import sys, {module}; assert not {banned!r} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", probe], check=True, cwd=ROOT)


def test_the_engine_and_services_import_nothing_else_of_the_package():
    allowed = ("assistanteval.world", "assistanteval.services")
    for path in [
        ROOT / "assistanteval/world.py",
        *(ROOT / "assistanteval/services").glob("*.py"),
    ]:
        blocked = _foreign(_imports(path), allowed)
        assert not blocked, f"{path.relative_to(ROOT)} imports {blocked}"


def _local_module(module: str) -> Path | None:
    base = ROOT.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _assistant_paths() -> set[Path]:
    """Every assistant adapter module and the local modules it imports."""
    pending = {
        path
        for path in (ROOT / "assistanteval/assistants").glob("*.py")
        if path.name != "__init__.py"
    }
    assert pending, "No assistant adapters found"
    seen = set()
    while pending:
        path = pending.pop()
        seen.add(path)
        pending.update(
            candidate
            for module in _imports(path)
            if (candidate := _local_module(module)) is not None
        )
        pending.difference_update(seen)
    return seen


def test_assistant_import_boundary():
    """Adapters drive assistants only: they reach no service code and no cloud.
    They may read their own model endpoint's settings."""
    for path in _assistant_paths():
        imports = _imports(path)
        allowed = ("assistanteval.contracts", "assistanteval.endpoints")
        blocked = _foreign(imports, allowed) | {
            module for module in imports if module.startswith("boto3")
        }
        assert not blocked, (
            f"{path.relative_to(ROOT)} crosses the assistant boundary: {blocked}"
        )


def test_assistant_boundary_follows_local_imports(tmp_path):
    adapters = tmp_path / "assistanteval/assistants"
    adapters.mkdir(parents=True)
    (adapters / "future.py").write_text("from custom import helpers\n")
    package = tmp_path / "custom/helpers"
    package.mkdir(parents=True)
    (tmp_path / "custom/__init__.py").write_text("")
    (package / "__init__.py").write_text("from assistanteval import world\n")
    with (
        patch(f"{__name__}.ROOT", tmp_path),
        pytest.raises(AssertionError, match="assistanteval.world"),
    ):
        test_assistant_import_boundary()


def test_every_built_in_assistant_is_a_chat_in_assistanteval_assistants():
    """`--assistant <name>` opens `open_chat` in assistanteval/assistants/<name>.py,
    whose adapter is <Name>Chat (model_baseline: ModelBaselineChat)."""
    import importlib

    from assistanteval.run_task import ASSISTANTS

    for name, entry in ASSISTANTS.items():
        assert entry == f"assistanteval.assistants.{name}:open_chat"
        module = importlib.import_module(f"assistanteval.assistants.{name}")
        chat = "".join(part.title() for part in name.split("_")) + "Chat"
        assert hasattr(module, chat), (
            f"assistanteval/assistants/{name}.py has no {chat}"
        )
        assert module.open_chat.connector_kind


def test_relative_contract_import_is_allowed(tmp_path):
    path = tmp_path / "assistanteval/assistants/future.py"
    path.parent.mkdir(parents=True)
    path.write_text("from ...assistanteval import contracts\n")
    with patch(f"{__name__}.ROOT", tmp_path):
        assert not _foreign(_imports(path), ("assistanteval.contracts",))
