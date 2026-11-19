from __future__ import annotations

from pathlib import Path

from src.orchestration.message import AgentMessage, make_message
from src.orchestration.task import AgentTask, now_iso
from src.utils.io import write_json


class RunState:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.tasks: list[AgentTask] = []
        self.messages: list[AgentMessage] = []
        self.run_log: list[dict] = []

    def start_task(self, agent_name: str, input_refs: list[str], retry_count: int = 0) -> AgentTask:
        task = AgentTask(
            task_id=f"TASK-{len(self.tasks) + 1:03d}",
            agent_name=agent_name,
            input_refs=input_refs,
            retry_count=retry_count,
            status="running",
        )
        self.tasks.append(task)
        self.log("task_started", {"task_id": task.task_id, "agent": agent_name})
        return task

    def finish_task(self, task: AgentTask, output_refs: list[str], status: str = "passed", issues: list[dict] | None = None) -> None:
        task.output_refs = output_refs
        task.status = status
        task.issues = issues or []
        task.finished_at = now_iso()
        self.log("task_finished", {"task_id": task.task_id, "status": status, "outputs": output_refs})

    def add_message(self, from_agent: str, to_agent: str, message_type: str, payload: dict, task_id: str) -> None:
        self.messages.append(make_message(from_agent, to_agent, message_type, payload, task_id))
        self.log("message", {"from": from_agent, "to": to_agent, "type": message_type})

    def log(self, event: str, payload: dict) -> None:
        self.run_log.append({"time": now_iso(), "event": event, "payload": payload})

    def flush(self) -> None:
        write_json(self.output_dir / "agent_tasks.json", self.tasks)
        write_json(self.output_dir / "agent_messages.json", self.messages)
        write_json(self.output_dir / "run_log.json", self.run_log)
