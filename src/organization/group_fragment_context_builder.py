from __future__ import annotations

from typing import Any

from src.contribution.schemas import GroupContext


class GroupFragmentContextBuilder:
    """Build the minimal, unit-first evidence card for group-level LLM tasks.

    Topic and group metadata belong to the enclosing task payload.  The card
    itself contains only the knowledge units that may support prose, avoiding
    duplicated group fields and keeping paper information within ``background``.
    """

    def build(self, group_context: GroupContext) -> dict[str, Any]:
        units: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for unit in group_context.units:
            paper_id = self._clean(unit.paper_id)
            background = self._clean(unit.background)
            if not paper_id:
                continue
            text = self._clean(unit.text)
            key = (paper_id, text)
            if not text or key in seen:
                continue
            seen.add(key)
            units.append(
                {
                    "text": text,
                    "background": {
                        "paper_id": paper_id,
                        "content": background,
                    },
                }
            )
        return {"units": units}

    def _clean(self, value: Any) -> str:
        return " ".join(str(value or "").split())
