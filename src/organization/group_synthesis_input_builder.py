from __future__ import annotations

from src.contribution.schemas import (
    ContributionRecord,
    GroupContext,
    GroupContextUnit,
    GroupContributionTable,
    KnowledgeUnit,
    UnitGroup,
)


class GroupContextBuilder:
    """Build the one shared, unit-first context for a group.

    ``UnitGroup.units`` remains the authoritative membership relation.  This
    class materializes it for LLM tasks, so an incrementally admitted unit is
    indistinguishable from an initial unit in both retrieval diagnosis and
    synthesis.
    """

    def build(
        self,
        review_topic: str,
        group: UnitGroup,
        records_by_id: dict[str, ContributionRecord],
        units_by_id: dict[str, KnowledgeUnit],
    ) -> GroupContext:
        units: list[GroupContextUnit] = []
        for unit_id in group.units:
            unit = units_by_id.get(unit_id)
            if unit is None:
                continue
            record = records_by_id.get(unit.paper_id)
            background = (record.paper_summary if record else unit.paper_summary) or unit.paper_id
            units.append(
                GroupContextUnit(
                    unit_id=unit.unit_id,
                    paper_id=unit.paper_id,
                    text=unit.text,
                    background=self._clean(background),
                )
            )
        return GroupContext(
            group_id=group.group_id,
            group_type=group.group_type,
            group_label=group.label,
            group_summary=group.summary,
            group_quality_flag=group.quality_flag,
            size=len(units),
            review_topic=review_topic,
            units=units,
            structure_scope=group.structure_scope,
            initial_size=group.initial_size if group.initial_size is not None else len(units),
            incremental_unit_count=group.incremental_unit_count,
            group_role=group.group_role,
        )

    def build_from_table(self, table: GroupContributionTable, review_topic: str = "") -> GroupContext:
        units = [
            GroupContextUnit(
                unit_id=unit.unit_id,
                paper_id=contribution.paper_id,
                text=unit.text,
                background=self._clean(contribution.paper_summary) or contribution.paper_id,
            )
            for contribution in table.contributions
            for unit in contribution.units
        ]
        return GroupContext(
            group_id=table.group_id,
            group_type=table.group_type,
            group_label=table.group_label,
            group_summary=table.group_summary,
            group_quality_flag=table.group_quality_flag,
            size=len(units),
            review_topic=review_topic,
            units=units,
            structure_scope="resumed_group_table",
            initial_size=len(units),
            group_role=table.group_role,
        )

    def _clean(self, value: object) -> str:
        return " ".join(str(value or "").split())


# Compatibility name for importers during the schema-v2 transition. New code
# must use GroupContextBuilder and group_contexts.json.
GroupSynthesisInputBuilder = GroupContextBuilder
