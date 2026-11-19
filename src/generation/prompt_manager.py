from __future__ import annotations

from pathlib import Path


class PromptManager:
    def __init__(self, prompt_dir: Path):
        self.prompt_dir = prompt_dir

    def load(self, name: str) -> str:
        path = self.prompt_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def load_many(self, names: list[str]) -> dict[str, str]:
        return {name: self.load(name) for name in names}
