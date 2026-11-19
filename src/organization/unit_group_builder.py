from __future__ import annotations

from statistics import mean

from src.contribution.schemas import ContributionRecord, KnowledgeUnit, UnitGroup, UnitRelation


def group_edge_weights(community: set[str], relations: list[UnitRelation]) -> list[float]:
    return [
        rel.similarity
        for rel in relations
        if rel.source_unit in community and rel.target_unit in community
    ]


def diagnose_group(
    size: int,
    density: float,
    mean_similarity: float,
    min_similarity: float,
    graph_threshold: float,
    paper_count: int | None = None,
    group_role: str = "semantic",
) -> str:
    if group_role == "other":
        return "other_group"
    effective_paper_count = paper_count if paper_count is not None else size
    if effective_paper_count == 1:
        return "singleton"
    if effective_paper_count == 2:
        return "small_group"
    if effective_paper_count > 10:
        return "oversized_group"
    if mean_similarity and mean_similarity < graph_threshold:
        return "low_confidence"
    return "normal"


class UnitGroupBuilder:
    def __init__(self, graph_threshold: float = 0.60):
        self.graph_threshold = graph_threshold

    def build(
        self,
        unit_type: str,
        communities: list[set[str]],
        modularity: float,
        units_by_id: dict[str, KnowledgeUnit],
        records_by_id: dict[str, ContributionRecord],
        relations: list[UnitRelation],
        algorithm: str,
        other_unit_ids: set[str] | None = None,
    ) -> list[UnitGroup]:
        groups: list[UnitGroup] = []
        regular_index = 0
        other_unit_ids = other_unit_ids or set()
        for community in communities:
            is_other = bool(community) and community <= other_unit_ids
            if is_other:
                group_id = f"{unit_type.upper()}G999"
            else:
                regular_index += 1
                group_id = f"{unit_type.upper()}G{regular_index:03d}"
            ordered_units = sorted(community)
            papers = sorted({units_by_id[unit_id].paper_id for unit_id in ordered_units})
            weights = group_edge_weights(community, relations)
            edge_count = len(weights)
            possible_edges = len(community) * (len(community) - 1) / 2
            density = round(edge_count / possible_edges, 4) if possible_edges else 0.0
            mean_similarity = round(mean(weights), 4) if weights else 0.0
            min_similarity = round(min(weights), 4) if weights else 0.0
            sample_texts = [units_by_id[uid].text for uid in ordered_units[:3]]
            group_role = "other" if is_other else "semantic"
            label = self._other_label(unit_type) if is_other else self._make_label(unit_type, sample_texts)
            if is_other:
                summary = f"{label}。该写作容器收纳 {len(ordered_units)} 个未形成稳定社区的单例知识单元。"
            else:
                summary = f"{label}。该组覆盖 {len(papers)} 篇候选文献：{', '.join(papers)}。"
            groups.append(
                UnitGroup(
                    group_id=group_id,
                    group_type=unit_type,
                    label=label,
                    units=ordered_units,
                    representative_papers=papers,
                    summary=summary,
                    size=len(community),
                    edge_count=edge_count,
                    density=density,
                    mean_similarity=mean_similarity,
                    min_similarity=min_similarity,
                    modularity=modularity,
                    community_algorithm="singleton_other_container" if is_other else algorithm,
                    quality_flag=diagnose_group(
                        len(community),
                        density,
                        mean_similarity,
                        min_similarity,
                        self.graph_threshold,
                        len(papers),
                        group_role,
                    ),
                    structure_scope="initial_grouping",
                    initial_size=len(community),
                    incremental_unit_count=0,
                    group_role=group_role,
                )
            )
        return groups

    def _other_label(self, unit_type: str) -> str:
        return {
            "issue": "其他研究议题",
            "problem": "其他研究问题",
            "method": "其他方法路径",
            "finding": "其他研究发现",
        }[unit_type]

    def _make_label(self, unit_type: str, sample_texts: list[str]) -> str:
        prefix = {
            "issue": "研究议题",
            "problem": "研究问题",
            "method": "方法路径",
            "finding": "研究发现",
        }[unit_type]
        text = sample_texts[0] if sample_texts else unit_type
        short = " ".join(text.split()[:8]).strip(" .")
        return f"{prefix}：{short}"
