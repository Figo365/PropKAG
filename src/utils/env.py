"""Small, dependency-free loader for project-local environment files."""

from __future__ import annotations

import os
from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    """Return simple ``KEY=VALUE`` entries from an environment file."""
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(path: Path) -> None:
    """Load simple ``KEY=VALUE`` entries without overriding process settings."""
    for key, value in read_env_file(path).items():
        os.environ.setdefault(key, value)


def load_project_env(project_root: Path) -> None:
    """Load the project's optional local configuration file.

    Values inherited from the shell are intentionally preserved, which makes
    the same configuration work locally, in CI, and in deployed environments.
    """
    load_env_file(project_root / ".env")
