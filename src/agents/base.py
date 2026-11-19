from __future__ import annotations

from pathlib import Path

from src.orchestration.run_state import RunState


class BaseAgent:
    name = "BaseAgent"

    def __init__(self, output_dir: Path, state: RunState):
        self.output_dir = output_dir
        self.state = state
