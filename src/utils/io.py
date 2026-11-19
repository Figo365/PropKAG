from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_text(path: str | Path, value: str) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(value, encoding="utf-8")
    return str(p)
