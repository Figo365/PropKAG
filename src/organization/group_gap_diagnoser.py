from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from src.contribution.schemas import GroupContext, GroupGap, ReviewFragment, UnitGroup
from src.organization.group_fragment_context_builder import GroupFragmentContextBuilder
from src.utils.llm_client import LLMClient


TYPE_PROFILES = {
    "issue": "只诊断议题范围、研究对象/场景、议题演进或议题间关系的缺口；需要补充的是 issue 单元。",
    "problem": "只诊断研究对象面临的具体难题、约束、评价目标或问题间关系的缺口；需要补充的是 research_problem 单元。",
    "method": "只诊断方法路线、方法假设、输入输出、方法比较、适用条件或局限的缺口；需要补充的是 research_method 单元。",
    "finding": "只诊断研究发现、结果模式、适用条件、相互矛盾的发现或结论边界的缺口；需要补充的是 research_finding 单元。",
}


class GroupGapDiagnoser:
    """Diagnose retrieval-worthy gaps from an already written group review.

    A configured LLM is required to open a retrieval gap. Without one, the
    system records diagnosis_failed rather than inventing a query or reporting
    that the group has no gap.
    """

    def __init__(self, prompts: dict[str, str] | None = None):
        self.prompts = prompts or {}
        self.llm_client = LLMClient()
        self.worker_count = self._worker_count("LLM_GAP_DIAGNOSIS_WORKERS", default=3)
        self.max_directions = min(3, self._worker_count("MAX_GAP_DIRECTIONS_PER_GROUP", default=3))

    def diagnose(
        self,
        review_topic: str,
        groups: list[UnitGroup],
        group_reviews: list[ReviewFragment],
        round_id: int,
        attempt_history: dict[str, list[dict]] | None = None,
        group_contexts: list[GroupContext] | None = None,
    ) -> tuple[list[GroupGap], dict]:
        attempt_history = attempt_history or {}
        reviews_by_id = {item.group_id: item for item in group_reviews}
        contexts_by_id = {item.group_id: item for item in group_contexts or []}
        def diagnose_group(group: UnitGroup) -> tuple[list[GroupGap], dict]:
            if group.group_role == "other":
                return [], {
                    "group_id": group.group_id,
                    "diagnosed": False,
                    "diagnosis_status": "excluded_other_group",
                    "reason": "other groups are excluded from iterative gap retrieval",
                }
            group_review = reviews_by_id.get(group.group_id)
            if group_review is None:
                return [], {
                    "group_id": group.group_id,
                    "diagnosed": False,
                    "diagnosis_status": "diagnosis_failed",
                    "reason": "missing_group_review",
                }
            group_attempts = attempt_history.get(group.group_id, [])
            data, mode, error = self._diagnose_one(
                review_topic, group, group_review, group_attempts, contexts_by_id.get(group.group_id)
            )
            previous_queries = {
                self._normalize_query(item.get("retrieval_query"))
                for item in group_attempts
                if item.get("retrieval_query")
            }
            gaps = self._to_gaps(data, group, round_id, previous_queries)
            diagnostic = {
                "group_id": group.group_id,
                "diagnosed": True,
                "diagnosis_status": self._diagnosis_status(data, mode, gaps),
                "mode": mode,
                "error": error,
                "results": [asdict(gap) for gap in gaps],
            }
            return [gap for gap in gaps if gap.need_retrieval], diagnostic

        # Each group reads immutable inputs and returns an isolated result.
        # map() retains group order when diagnostics are merged.
        with ThreadPoolExecutor(max_workers=min(self.worker_count, max(1, len(groups)))) as executor:
            outcomes = list(executor.map(diagnose_group, groups))
        gaps = [gap for group_gaps, _ in outcomes for gap in group_gaps]
        diagnostics = [diagnostic for _, diagnostic in outcomes]
        return gaps, {
            "round_id": round_id,
            "llm_enabled": self.llm_client.enabled,
            "worker_count": min(self.worker_count, len(groups)),
            "max_directions_per_group": self.max_directions,
            "failed_group_count": sum(item.get("diagnosis_status") == "diagnosis_failed" for item in diagnostics),
            "groups": diagnostics,
        }

    @staticmethod
    def _worker_count(variable: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(variable) or default))
        except ValueError:
            return default

    def _diagnose_one(
        self,
        review_topic: str,
        group: UnitGroup,
        group_review: ReviewFragment,
        previous_attempts: list[dict] | None = None,
        group_context: GroupContext | None = None,
    ) -> tuple[dict, str, str]:
        prompt = self.prompts.get("group_iterative_gap_diagnosis")
        if self.llm_client.enabled and prompt:
            try:
                result = self.llm_client.chat_json(
                    prompt,
                    {
                        "review_topic": review_topic,
                        "target_group_type": group.group_type,
                        "type_specific_diagnosis_rule": TYPE_PROFILES.get(group.group_type, ""),
                        "group_assessment_input": self._payload(group, group_review, group_context),
                        "previous_retrieval_attempts": previous_attempts or [],
                    },
                    temperature=0.0,
                )
                if not isinstance(result, dict):
                    raise ValueError("Gap diagnosis must return a JSON object.")
                return result, "llm", ""
            except Exception as exc:
                fallback = self._fallback()
                return fallback, "safe_closed_fallback", str(exc)
        return self._fallback(), "llm_unavailable", ""

    def _payload(self, group: UnitGroup, review: ReviewFragment, group_context: GroupContext | None = None) -> dict:
        return {
            "synthesis_context_card": (
                GroupFragmentContextBuilder().build(group_context)
                if group_context is not None
                else {
                    "units": [],
                }
            ),
            "group": {
                "group_id": group.group_id,
                "group_type": group.group_type,
                "group_label": group.label,
                "quality_flag": group.quality_flag,
                "paper_count": len(group.representative_papers),
                "structure_scope": group.structure_scope,
                "incremental_unit_count": group.incremental_unit_count,
            },
            "review": {
                "fragment_id": review.fragment_id,
                "title": review.title,
                "text": review.text,
                "cited_papers": review.cited_papers,
            },
        }

    def _diagnosis_status(self, data: dict, mode: str, gaps: list[GroupGap]) -> str:
        if mode in {"safe_closed_fallback", "llm_unavailable"}:
            return "diagnosis_failed"
        if not isinstance(data, dict):
            return "diagnosis_failed"
        decision = self._decision(data)
        if decision == "retrieval_required":
            if not gaps:
                return "diagnosis_failed"
            if all("repeated_query_rejected" in gap.description for gap in gaps):
                return "query_rejected"
            return "gap_found" if any(gap.need_retrieval for gap in gaps) else "diagnosis_failed"
        if decision == "rewrite_required":
            return "rewrite_required" if self._clean(data.get("gap_description") or data.get("reason")) else "diagnosis_failed"
        if decision == "sufficient":
            return "no_gap"
        return "diagnosis_failed"

    def _fallback(self) -> dict:
        return {
            "decision": "sufficient",
            "gap_description": "LLM 未配置或诊断失败；不生成未经 LLM 判断的补充检索。",
            "retrieval_query": "",
            "gaps": [],
        }

    def _to_gaps(
        self,
        data: dict,
        group: UnitGroup,
        round_id: int,
        previous_queries: set[str] | None = None,
    ) -> list[GroupGap]:
        """Parse up to N independent retrieval directions for one group.

        The legacy single-gap shape remains accepted so prior prompt outputs and
        direct callers continue to work.
        """
        raw_gaps = data.get("gaps") if isinstance(data, dict) else None
        if not isinstance(raw_gaps, list):
            return [self._to_gap(data, group, round_id, previous_queries, gap_index=1)]
        if self._decision(data) == "sufficient":
            return [self._to_gap(data, group, round_id, previous_queries, gap_index=1)]

        gaps: list[GroupGap] = []
        seen_queries = set(previous_queries or set())
        for item in raw_gaps[: self.max_directions]:
            if not isinstance(item, dict):
                continue
            candidate = {
                "decision": "retrieval_required",
                "gap_description": item.get("gap_description") or item.get("description") or item.get("direction"),
                "retrieval_query": item.get("retrieval_query"),
            }
            gap = self._to_gap(candidate, group, round_id, seen_queries, gap_index=len(gaps) + 1)
            if gap.need_retrieval:
                seen_queries.add(self._normalize_query(gap.retrieval_query))
            gaps.append(gap)
        return gaps

    def _to_gap(
        self,
        data: dict,
        group: UnitGroup,
        round_id: int,
        previous_queries: set[str] | None = None,
        gap_index: int = 1,
    ) -> GroupGap:
        decision = self._decision(data)
        requested_retrieval = decision == "retrieval_required"
        description = self._clean(data.get("gap_description") or data.get("description") or data.get("reason"))
        retrieval_query = self._clean(data.get("retrieval_query"))
        need_retrieval = requested_retrieval and bool(description) and bool(retrieval_query)
        if need_retrieval and self._normalize_query(retrieval_query) in (previous_queries or set()):
            return GroupGap(
                gap_id=f"GAP-R{round_id:02d}-{group.group_id}-{gap_index:02d}",
                round_id=round_id,
                group_id=group.group_id,
                group_type=group.group_type,
                need_retrieval=False,
                description=f"{description}（repeated_query_rejected：该组已尝试完全相同的检索 query。）",
                retrieval_query="",
                decision="sufficient",
            )
        if requested_retrieval and not need_retrieval:
            detail = description or "未提供缺口描述"
            description = f"诊断输出无效：LLM 判定需要检索，但未同时提供完整的缺口描述和检索 query。原描述：{detail}"
            retrieval_query = ""
        elif not requested_retrieval:
            description = description or "LLM 判定当前组不需要补充检索。"
            retrieval_query = ""
        return GroupGap(
            gap_id=f"GAP-R{round_id:02d}-{group.group_id}-{gap_index:02d}",
            round_id=round_id,
            group_id=group.group_id,
            group_type=group.group_type,
            need_retrieval=need_retrieval,
            description=description,
            retrieval_query=retrieval_query,
            decision=decision,
        )

    def _decision(self, data: dict) -> str:
        value = self._clean(data.get("decision")).lower()
        if value in {"sufficient", "rewrite_required", "retrieval_required"}:
            return value
        # Compatibility with persisted prompt outputs from schema-v3 runs.
        if isinstance(data.get("need_retrieval"), bool):
            return "retrieval_required" if data["need_retrieval"] else "sufficient"
        return ""

    def _clean(self, value) -> str:
        return " ".join(str(value or "").split())

    def _normalize_query(self, value) -> str:
        return self._clean(value).casefold().rstrip(".?!。？！")
