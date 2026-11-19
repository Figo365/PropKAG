from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class AgentMessage:
    message_id: str
    from_agent: str
    to_agent: str
    message_type: str
    payload: dict
    related_task_id: str


def make_message(from_agent: str, to_agent: str, message_type: str, payload: dict, task_id: str) -> AgentMessage:
    return AgentMessage(
        message_id=f"MSG-{uuid4().hex[:10]}",
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        payload=payload,
        related_task_id=task_id,
    )
