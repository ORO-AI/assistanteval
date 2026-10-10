"""The repository is published: tracked files must not carry private data."""

import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parent.parent
PRIVATE = (
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",  # a private key
    r"(^|[^A-Za-z0-9-])sk-[A-Za-z0-9_-]{32,}",  # an OpenAI-style API key
    r"AKIA[0-9A-Z]{16}",  # an AWS access key id
    r"(/Users|/home)/[A-Za-z]",  # a path in someone's home directory
)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def test_no_env_file_is_tracked():
    tracked = git("ls-files").stdout.splitlines()
    env = [
        path
        for path in tracked
        if pathlib.PurePath(path).name.startswith(".env")
        and not path.endswith(".example")
    ]
    assert env == []


def test_tracked_files_carry_no_keys_or_home_paths():
    """The pathspec excludes this file, which necessarily holds the patterns."""
    hits = git(
        "grep",
        "-I",
        "-n",
        "-E",
        "-e",
        "|".join(PRIVATE),
        "--",
        f":!{HERE.relative_to(ROOT)}",
    )
    # returncode 1 is git grep's "no matches", which is the passing case.
    assert hits.returncode == 1, f"private data in tracked files:\n{hits.stdout}"
