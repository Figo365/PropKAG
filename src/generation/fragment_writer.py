from __future__ import annotations

from src.contribution.schemas import GroupContributionTable, ReviewFragment


TYPE_NAMES = {
    "issue": "研究议题",
    "problem": "核心问题",
    "method": "方法路径",
    "finding": "研究发现",
}


class FragmentWriter:
    def write(self, table: GroupContributionTable) -> ReviewFragment:
        papers = [contribution.paper_id for contribution in table.contributions]
        if table.group_role == "other":
            selected = table.contributions[:5]
            mentions = []
            for contribution in selected:
                unit_text = contribution.units[0].text if contribution.units else "一项补充方向"
                mentions.append(f"“{unit_text}”（{contribution.paper_id}）")
            text = "除上述主题外，候选文献还涉及" + "、".join(mentions) + "。"
            return ReviewFragment(
                fragment_id=f"FRAG-{table.group_id}",
                group_id=table.group_id,
                group_type=table.group_type,
                title=table.group_label,
                text=text,
                cited_papers=[contribution.paper_id for contribution in selected],
                group_role="other",
            )
        if table.group_type == "issue":
            representative_papers = list(dict.fromkeys(papers))[:4]
            citation = f"（{', '.join(representative_papers)}）" if representative_papers else ""
            qualifier = "单篇研究关注" if len(representative_papers) == 1 else "现有摘要级知识单元共同关注"
            text = f"{qualifier}“{table.group_label}”这一检索主题侧面{citation}。"
            return ReviewFragment(
                fragment_id=f"FRAG-{table.group_id}",
                group_id=table.group_id,
                group_type=table.group_type,
                title=table.group_label,
                text=text,
                cited_papers=representative_papers,
                group_role=table.group_role,
            )
        cited = "、".join(papers)
        caution = "相关研究关注" if table.group_quality_flag != "normal" else "共同关注"
        type_name = TYPE_NAMES[table.group_type]
        first = table.contributions[0] if table.contributions else None
        target_hint = first.units[0].text if first and first.units else ""
        text = f"{table.group_label}体现了候选文献中的{caution}，支撑论文包括 {cited}。"
        if target_hint:
            text += f" 其中，当前组的直接知识单元包括“{target_hint}”（{papers[0]}）。"
        return ReviewFragment(
            fragment_id=f"FRAG-{table.group_id}",
            group_id=table.group_id,
            group_type=table.group_type,
            title=table.group_label,
            text=text,
            cited_papers=papers,
            group_role=table.group_role,
        )
