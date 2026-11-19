from __future__ import annotations

from dataclasses import replace
import re

from src.contribution.schemas import ContributionRecord, GroupContext, GroupContributionTable, GroupIntegration, KnowledgeUnit, Paper, UnitGroup
from src.organization.group_synthesis_input_builder import GroupContextBuilder
from src.organization.group_table_builder import GroupContributionTableBuilder
from src.utils.paper_identity import paper_identity_keys


FIELD_BY_TYPE = {
    "issue": "issues",
    "problem": "research_problems",
    "method": "research_methods",
    "finding": "research_findings",
}
CODE_BY_TYPE = {"issue": "I", "problem": "P", "method": "M", "finding": "F"}


class GroupIncrementalIntegrator:
    """Append an accepted target-type unit to its fixed target group only."""

    def integrate(
        self,
        accepted: list[dict],
        papers: list[Paper],
        records: list[ContributionRecord],
        units: list[KnowledgeUnit],
        groups: list[UnitGroup],
        tables: list[GroupContributionTable],
        group_contexts: list[GroupContext],
        review_topic: str,
        round_id: int,
    ) -> tuple[list[Paper], list[ContributionRecord], list[KnowledgeUnit], list[UnitGroup], list[GroupContributionTable], list[GroupContext], list[GroupIntegration]]:
        groups_by_id = {group.group_id: group for group in groups}
        records_by_id = {record.paper_id: record for record in records}
        units_by_id = {unit.unit_id: unit for unit in units}
        tables_by_id = {table.group_id: table for table in tables}
        contexts_by_id = {item.group_id: item for item in group_contexts}
        papers_by_key = {key: paper for paper in papers for key in paper_identity_keys(paper)}
        next_id = self._next_paper_number(papers)
        integrations: list[GroupIntegration] = []

        for index, item in enumerate(accepted, start=1):
            gap = item["gap"]
            group = groups_by_id.get(gap.group_id)
            if group is None or group.group_type != gap.group_type:
                continue
            decision = item["decision"]
            candidate = item["paper"]
            candidate_keys = paper_identity_keys(candidate)
            paper = next((papers_by_key[key] for key in candidate_keys if key in papers_by_key), None)
            reused_paper = paper is not None
            if paper is None:
                paper = replace(
                    candidate,
                    paper_id=f"P{next_id:03d}",
                    retrieval_round=round_id,
                    gap_id=gap.gap_id,
                    admission_reason=decision["reason"],
                    source="iterative_retrieval",
                )
                next_id += 1
                papers.append(paper)
                for key in paper_identity_keys(paper):
                    papers_by_key[key] = paper

            record = records_by_id.get(paper.paper_id)
            if record is None:
                record = self._target_record(paper, group.group_type, decision["unit_text"], decision["source_evidence"])
                records.append(record)
                records_by_id[record.paper_id] = record
            else:
                self._append_target_unit(record, group.group_type, decision["unit_text"], decision["source_evidence"])

            unit = KnowledgeUnit(
                unit_id=self._next_unit_id(paper.paper_id, group.group_type, units),
                paper_id=paper.paper_id,
                unit_type=group.group_type,
                text=decision["unit_text"],
                evidence_text=decision["source_evidence"],
                paper_summary=record.paper_summary,
            )
            units.append(unit)
            units_by_id[unit.unit_id] = unit
            group.units.append(unit.unit_id)
            if paper.paper_id not in group.representative_papers:
                group.representative_papers.append(paper.paper_id)
            group.size = len(group.units)
            group.structure_scope = "initial_grouping_with_incremental_content"
            group.initial_size = group.initial_size if group.initial_size is not None else max(group.size - 1, 0)
            group.incremental_unit_count += 1
            group.summary = self._updated_summary(group.summary, decision["unit_text"])
            tables_by_id[group.group_id] = GroupContributionTableBuilder().build(group, records_by_id, units_by_id)
            # Rebuild from the authoritative group membership after every
            # admission. This makes the new unit visible to both the next gap
            # diagnosis and the eventual group synthesis.
            contexts_by_id[group.group_id] = GroupContextBuilder().build(
                review_topic, group, records_by_id, units_by_id
            )
            integrations.append(
                GroupIntegration(
                    integration_id=f"INT-R{round_id:02d}-{index:03d}",
                    round_id=round_id,
                    gap_id=gap.gap_id,
                    target_group_id=group.group_id,
                    paper_id=paper.paper_id,
                    query_id=str(candidate.query_id or ""),
                    reused_paper=reused_paper,
                    unit_type=group.group_type,
                    unit_text=unit.text,
                    source_evidence=unit.evidence_text or "",
                    integration_role=decision["integration_role"],
                    decision_reason=decision["reason"],
                )
            )
        return papers, records, units, groups, list(tables_by_id.values()), list(contexts_by_id.values()), integrations

    def _target_record(self, paper: Paper, group_type: str, unit_text: str, evidence: str) -> ContributionRecord:
        fields = {"issues": [], "research_problems": [], "research_methods": [], "research_findings": []}
        fields[FIELD_BY_TYPE[group_type]] = [unit_text]
        return ContributionRecord(
            paper_id=paper.paper_id,
            evidence_text={"title": paper.title, "abstract": paper.abstract, "keywords": "; ".join(paper.keywords), FIELD_BY_TYPE[group_type]: evidence},
            paper_summary=self._paper_summary(paper),
            **fields,
        )

    def _append_target_unit(self, record: ContributionRecord, group_type: str, unit_text: str, evidence: str) -> None:
        field_name = FIELD_BY_TYPE[group_type]
        values = getattr(record, field_name)
        if unit_text not in values:
            values.append(unit_text)
        current_evidence = " ".join(str(record.evidence_text.get(field_name, "")).split())
        evidence = " ".join(str(evidence or "").split())
        if evidence and evidence not in current_evidence:
            record.evidence_text[field_name] = f"{current_evidence}；{evidence}" if current_evidence else evidence

    def _next_unit_id(self, paper_id: str, group_type: str, units: list[KnowledgeUnit]) -> str:
        prefix = f"{paper_id}_{CODE_BY_TYPE[group_type]}"
        numbers = []
        for unit in units:
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", unit.unit_id)
            if match:
                numbers.append(int(match.group(1)))
        return f"{prefix}{max(numbers, default=0) + 1:02d}"

    def _paper_summary(self, paper: Paper) -> str:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", paper.abstract) if part.strip()]
        return sentences[0] if sentences else paper.title

    def _next_paper_number(self, papers: list[Paper]) -> int:
        values = [int(match.group(1)) for paper in papers if (match := re.fullmatch(r"P(\d+)", paper.paper_id))]
        return max(values, default=0) + 1

    def _updated_summary(self, current: str, unit_text: str) -> str:
        current = " ".join(str(current or "").split())
        unit_text = " ".join(str(unit_text or "").split())
        if not unit_text or unit_text in current:
            return current
        if not current:
            return unit_text
        return f"{current}；迭代检索补充：{unit_text}"
