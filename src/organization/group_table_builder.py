from __future__ import annotations

from src.contribution.schemas import ContributionRecord, GroupContribution, GroupContributionTable, GroupUnitRef, KnowledgeUnit, UnitGroup


FIELD_BY_TYPE = {
    "issue": "issues",
    "problem": "research_problems",
    "method": "research_methods",
    "finding": "research_findings",
}


class GroupContributionTableBuilder:
    def build(
        self,
        group: UnitGroup,
        records_by_id: dict[str, ContributionRecord],
        units_by_id: dict[str, KnowledgeUnit],
    ) -> GroupContributionTable:
        contributions = []
        for paper_id in group.representative_papers:
            record = records_by_id.get(paper_id)
            if record is None:
                continue
            target_units = [
                units_by_id[unit_id]
                for unit_id in group.units
                if unit_id in units_by_id and units_by_id[unit_id].paper_id == paper_id
            ]
            if not target_units:
                continue
            contributions.append(
                GroupContribution(
                    paper_id=paper_id,
                    paper_summary=record.paper_summary,
                    units=[
                        GroupUnitRef(unit_id=unit.unit_id, unit_type=unit.unit_type, text=unit.text)
                        for unit in target_units
                    ],
                )
            )
        return GroupContributionTable(
            group_id=group.group_id,
            group_type=group.group_type,
            group_label=group.label,
            group_summary=group.summary,
            group_quality_flag=group.quality_flag,
            contributions=contributions,
            group_role=group.group_role,
        )
