from __future__ import annotations

from src.contribution.schemas import ReviewFragment, TypeSynthesis, TypeSynthesisParagraph, TypeSynthesisSentence


TYPE_TITLES = {
    "issue": "研究关注与议题分布",
    "problem": "核心研究问题",
    "method": "主要方法路径",
    "finding": "研究发现与认识",
}


class TypeSynthesisIntegrator:
    """Fail-closed type integration fallback.

    A title-based pseudo-summary is worse than an explicit unavailable marker:
    it creates claims that no source fragment actually supports.  Normal runs
    use the LLM integrator; this class is only the honest last resort.
    """

    def integrate(self, group_type: str, fragments: list[ReviewFragment]) -> TypeSynthesis:
        if len(fragments) > 1:
            return self._unavailable(group_type, fragments)
        paragraphs = [TypeSynthesisParagraph(
            relation_focus=fragment.title,
            text=fragment.text.strip(),
            source_fragment_ids=[fragment.fragment_id],
            cited_papers=sorted(set(fragment.cited_papers)),
            role="synthesis",
            sentences=[TypeSynthesisSentence(
                text=fragment.text.strip(),
                source_fragment_ids=[fragment.fragment_id],
                cited_papers=sorted(set(fragment.cited_papers)),
            )],
        ) for fragment in fragments if fragment.text.strip()]
        return TypeSynthesis(
            type_synthesis_id=f"TYPE-{group_type.upper()}",
            group_type=group_type,
            title=TYPE_TITLES[group_type],
            text="\n\n".join(paragraph.text for paragraph in paragraphs),
            cited_papers=sorted({paper_id for fragment in fragments for paper_id in fragment.cited_papers}),
            covered_fragment_ids=[fragment.fragment_id for fragment in fragments],
            uncovered_fragment_ids=[],
            paragraphs=paragraphs,
            quality_issues=[],
        )

    def _unavailable(self, group_type: str, fragments: list[ReviewFragment]) -> TypeSynthesis:
        text = "该类型的自动整合未通过结构化质量验收，未将不可靠的自动文本写入终稿。"
        return TypeSynthesis(
            type_synthesis_id=f"TYPE-{group_type.upper()}",
            group_type=group_type,
            title=TYPE_TITLES[group_type],
            text=text,
            cited_papers=[],
            covered_fragment_ids=[],
            uncovered_fragment_ids=[fragment.fragment_id for fragment in fragments],
            paragraphs=[],
            quality_issues=[
                {
                    "issue_type": "type_synthesis_unavailable",
                    "group_type": group_type,
                    "reason": (
                        "Type-level LLM output failed hard validation. The system withheld a title-based "
                        "pseudo-summary rather than emitting unsupported prose."
                    ),
                }
            ],
        )
