from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

from src.contribution.schemas import GroupContext, GroupGap, Paper
from src.utils.llm_client import LLMClient
from src.utils.paper_identity import paper_identity_keys


class GroupCandidateFilter:
    """Decide candidate admission and extract one gap-filling unit in one LLM call."""

    def __init__(self, prompts: dict[str, str] | None = None):
        self.prompts = prompts or {}
        self.llm_client = LLMClient()
        self.worker_count = self._worker_count("LLM_CANDIDATE_FILTER_WORKERS", default=4)

    def filter_and_extract(
        self,
        candidates: list[dict],
        gaps: list[GroupGap],
        group_contexts: list[GroupContext],
        existing_papers: list[Paper],
    ) -> tuple[list[dict], list[dict]]:
        gaps_by_id = {gap.gap_id: gap for gap in gaps}
        contexts_by_id = {item.group_id: item for item in group_contexts}
        papers_by_id = {paper.paper_id: paper for paper in existing_papers}
        existing_keys_by_group = {
            item.group_id: {
                key
                for paper_id in self._context_paper_ids(item)
                if paper_id in papers_by_id
                for key in paper_identity_keys(papers_by_id[paper_id])
            }
            for item in group_contexts
        }
        prepared: list[tuple[GroupGap, GroupContext, Paper, set[str]] | None] = []
        prechecked: list[dict | None] = []
        queued_keys_by_group: dict[str, set[str]] = {group_id: set(keys) for group_id, keys in existing_keys_by_group.items()}
        for candidate in candidates:
            gap_id = str(candidate.get("gap_id") or "")
            group_id = str(candidate.get("group_id") or "")
            gap = gaps_by_id.get(gap_id)
            group_context = contexts_by_id.get(group_id)
            try:
                paper_data = candidate.get("paper", {})
                if not isinstance(paper_data, dict):
                    raise TypeError("paper must be a JSON object")
                paper = Paper(**{key: value for key, value in paper_data.items() if key in Paper.__dataclass_fields__})
            except (TypeError, ValueError) as exc:
                prepared.append(None)
                prechecked.append({"gap_id": gap_id, "title": "", "reason": f"invalid_candidate_paper: {exc}"})
                continue
            if gap is None or group_context is None:
                prepared.append(None)
                prechecked.append({"gap_id": gap_id, "title": paper.title, "reason": "missing_target_group_context"})
                continue
            group_existing_keys = queued_keys_by_group.setdefault(group_id, set())
            paper_keys = set(paper_identity_keys(paper))
            if not paper_keys:
                prepared.append(None)
                prechecked.append({"gap_id": gap.gap_id, "title": paper.title, "reason": "missing_paper_identity"})
                continue
            if paper_keys & group_existing_keys:
                prepared.append(None)
                prechecked.append({"gap_id": gap.gap_id, "title": paper.title, "reason": "duplicate_with_target_group"})
                continue
            group_existing_keys.update(paper_keys)
            prepared.append((gap, group_context, paper, paper_keys))
            prechecked.append(None)

        def decide(prepared_item: tuple[GroupGap, GroupContext, Paper, set[str]] | None) -> dict | None:
            if prepared_item is None:
                return None
            gap, group_context, paper, _ = prepared_item
            return self._decide(gap, group_context, paper)

        with ThreadPoolExecutor(max_workers=min(self.worker_count, max(1, len(candidates)))) as executor:
            decisions = list(executor.map(decide, prepared))

        accepted: list[dict] = []
        rejected: list[dict] = []
        for prepared_item, immediate_rejection, decision in zip(prepared, prechecked, decisions):
            if immediate_rejection is not None:
                rejected.append(immediate_rejection)
                continue
            assert prepared_item is not None and decision is not None
            gap, _, paper, _ = prepared_item
            if decision["decision"] != "accept":
                rejected.append({"gap_id": gap.gap_id, "title": paper.title, "reason": decision["reason"]})
                continue
            accepted.append({"gap": gap, "paper": paper, "decision": decision})
        return accepted, rejected

    @staticmethod
    def _worker_count(variable: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(variable) or default))
        except ValueError:
            return default

    def _decide(self, gap: GroupGap, group_context: GroupContext, paper: Paper) -> dict:
        prompt = self.prompts.get("group_gap_filter_and_extract")
        if self.llm_client.enabled and prompt:
            try:
                result = self.llm_client.chat_json(
                    prompt,
                    {
                        "gap": gap.__dict__,
                        "target_group": {
                            "group_id": group_context.group_id,
                            "group_type": group_context.group_type,
                            "group_label": group_context.group_label,
                            "group_summary": group_context.group_summary,
                            "quality_flag": group_context.group_quality_flag,
                        },
                        "target_units": [
                            {
                                "unit_id": unit.unit_id,
                                "text": unit.text,
                                "background": {"paper_id": unit.paper_id, "content": unit.background},
                            }
                            for unit in group_context.units
                        ],
                        "candidate_paper": {"title": paper.title, "abstract": paper.abstract, "keywords": paper.keywords},
                    },
                    temperature=0.0,
                )
                return self._normalize_decision(result, paper)
            except Exception as exc:
                return {"decision": "reject", "reason": f"LLM filter failed; candidate rejected; error={exc}"}
        return {"decision": "reject", "reason": "LLM filter unavailable; candidate rejected to avoid unsupported admission."}

    def _normalize_decision(self, result: dict, paper: Paper) -> dict:
        decision = str(result.get("decision") or "reject").lower()
        unit_text = self._clean(result.get("target_unit_text"))
        evidence = self._clean(result.get("source_evidence"))
        if decision != "accept" or not unit_text or not evidence:
            return {"decision": "reject", "reason": self._clean(result.get("reason")) or "未提供可融入目标组的摘要级单元与证据。"}
        if not self._evidence_is_in_paper(evidence, paper):
            return {"decision": "reject", "reason": "source_evidence 不是候选论文题名、摘要或关键词中的可核对原文。"}
        return {
            "decision": "accept",
            "unit_text": unit_text,
            "source_evidence": evidence,
            "integration_role": self._clean(result.get("integration_role")) or "回应目标组缺口",
            "reason": self._clean(result.get("reason")) or "候选论文直接回应目标组缺口。",
        }

    def _context_paper_ids(self, group_context: GroupContext) -> set[str]:
        return {unit.paper_id for unit in group_context.units if unit.paper_id}

    def _evidence_is_in_paper(self, evidence: str, paper: Paper) -> bool:
        source = " ".join([paper.title, paper.abstract, " ".join(paper.keywords)])
        return self._normalize_for_match(evidence) in self._normalize_for_match(source)

    def _normalize_for_match(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" \"'“”‘’").casefold()

    def _clean(self, value) -> str:
        return " ".join(str(value or "").split())
