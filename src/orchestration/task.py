from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentTask:
    task_id: str
    agent_name: str
    input_refs: list[str]
    output_refs: list[str] = field(default_factory=list)
    status: str = "pending"
    issues: list[dict] = field(default_factory=list)
    retry_count: int = 0
    created_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
