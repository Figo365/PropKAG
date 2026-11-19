from __future__ import annotations

from src.contribution.schemas import ContributionRecord, KnowledgeUnit


TYPE_TO_FIELD = {
    "issue": "issues",
    "problem": "research_problems",
    "method": "research_methods",
    "finding": "research_findings",
}

TYPE_CODE = {
    "issue": "I",
    "problem": "P",
    "method": "M",
    "finding": "F",
}

# A paper contributes at most one core unit for each contribution type.  The
# abstract remains available in the evidence store; the unit is the compact
# grouping representation used by organization and synthesis.
MAX_UNITS_PER_TYPE = 1


class UnitIndexer:
    def expand(self, records: list[ContributionRecord]) -> list[KnowledgeUnit]:
        units: list[KnowledgeUnit] = []
        for record in records:
            for unit_type, field_name in TYPE_TO_FIELD.items():
                values = getattr(record, field_name)
                for idx, text in enumerate(values[:MAX_UNITS_PER_TYPE], start=1):
                    evidence = record.evidence_text.get(field_name, record.evidence_text.get("abstract"))
                    units.append(
                        KnowledgeUnit(
                            unit_id=f"{record.paper_id}_{TYPE_CODE[unit_type]}{idx:02d}",
                            paper_id=record.paper_id,
                            unit_type=unit_type,
                            text=text,
                            evidence_text=evidence,
                            paper_summary=record.paper_summary,
                        )
                    )
        return units
