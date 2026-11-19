from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from src.contribution.schemas import KnowledgeUnit, UnitGroup
from src.utils.llm_client import LLMClient


class UnitGroupNamer:
    """Name groups in small LLM batches while preserving rule-based fallbacks."""

    def __init__(self, prompts: dict[str, str] | None = None):
        self.prompts = prompts or {}
        self.llm_client = LLMClient.from_env("LLM_BASIC")
        self.batch_size = max(1, int(os.getenv("LLM_GROUP_NAMING_BATCH_SIZE", "8")))
        self.worker_count = self._worker_count("LLM_GROUP_NAMING_WORKERS", default=3)

    def name(
        self,
        groups: list[UnitGroup],
        units_by_id: dict[str, KnowledgeUnit],
        review_topic: str,
    ) -> tuple[list[UnitGroup], dict]:
        prompt = self.prompts.get("unit_group_naming")
        semantic_groups = [group for group in groups if group.group_role != "other"]
        statuses: list[dict] = [
            {
                "group_id": group.group_id,
                "mode": "deterministic_other_group",
                "error": "",
            }
            for group in groups
            if group.group_role == "other"
        ]
        if not self.llm_client.enabled or not prompt:
            reason = "LLM unavailable" if not self.llm_client.enabled else "unit_group_naming prompt missing"
            statuses.extend(
                {"group_id": group.group_id, "mode": "rule_fallback", "error": reason}
                for group in semantic_groups
            )
            return groups, {
                "mode": "rule_fallback",
                "llm_enabled": self.llm_client.enabled,
                "groups": statuses,
            }

        batches = [
            semantic_groups[start : start + self.batch_size]
            for start in range(0, len(semantic_groups), self.batch_size)
        ]

        def name_batch(batch: list[UnitGroup]) -> list[dict]:
            batch_statuses: list[dict] = []
            try:
                result = LLMClient.from_env("LLM_BASIC").chat_json(
                    prompt,
                    {
                        "review_topic": review_topic,
                        "groups": [self._payload(group, units_by_id) for group in batch],
                    },
                    temperature=0.0,
                )
                rows = result.get("groups") if isinstance(result, dict) else None
                if not isinstance(rows, list):
                    raise ValueError("response.groups must be a list")
                rows_by_id = {
                    str(row.get("group_id")): row
                    for row in rows
                    if isinstance(row, dict) and row.get("group_id")
                }
                for group in batch:
                    row = rows_by_id.get(group.group_id)
                    error = self._apply(group, row)
                    batch_statuses.append(
                        {
                            "group_id": group.group_id,
                            "mode": "llm" if not error else "rule_fallback",
                            "error": error,
                        }
                    )
                return batch_statuses
            except Exception as exc:
                return [
                    {"group_id": group.group_id, "mode": "rule_fallback", "error": str(exc)}
                    for group in batch
                ]

        # Batches operate on distinct groups, so calls can run concurrently;
        # executor.map preserves the original batch order when statuses merge.
        with ThreadPoolExecutor(max_workers=min(self.worker_count, max(1, len(batches)))) as executor:
            batch_statuses = list(executor.map(name_batch, batches))
        for statuses_for_batch in batch_statuses:
            statuses.extend(statuses_for_batch)
        modes = {item["mode"] for item in statuses if item["mode"] != "deterministic_other_group"}
        overall_mode = (
            "deterministic_other_group"
            if not modes
            else next(iter(modes)) if len(modes) == 1 else "mixed"
        )
        return groups, {
            "mode": overall_mode,
            "llm_profile": "LLM_BASIC",
            "llm_enabled": True,
            "model": self.llm_client.model,
            "batch_size": self.batch_size,
            "worker_count": min(self.worker_count, len(batches)),
            "groups": statuses,
        }

    @staticmethod
    def _worker_count(variable: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(variable) or default))
        except ValueError:
            return default

    def _payload(self, group: UnitGroup, units_by_id: dict[str, KnowledgeUnit]) -> dict:
        data = asdict(group)
        data["unit_texts"] = [units_by_id[unit_id].text for unit_id in group.units if unit_id in units_by_id]
        return data

    def _apply(self, group: UnitGroup, row: dict | None) -> str:
        if not row:
            return "missing group result"
        label = " ".join(str(row.get("label") or "").split())
        summary = " ".join(str(row.get("summary") or "").split())
        if not label or not summary:
            return "label or summary is empty"
        if len(label) > 80:
            return "label is not a short phrase"
        mentioned_papers = set(re.findall(r"\bP\d+\b", f"{label} {summary}", flags=re.IGNORECASE))
        allowed_papers = {paper_id.upper() for paper_id in group.representative_papers}
        if {paper_id.upper() for paper_id in mentioned_papers} - allowed_papers:
            return "result mentions a paper outside the group"
        group.label = label
        group.summary = summary
        return ""
