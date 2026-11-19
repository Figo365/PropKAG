from __future__ import annotations

from src.contribution.schemas import TypeSynthesis


class FinalReviewIntegrator:
    def integrate(self, review_topic: str, type_syntheses: list[TypeSynthesis]) -> str:
        by_type = {item.group_type: item for item in type_syntheses}

        # The review title is intentionally deferred until a dedicated title
        # generation stage is introduced. Do not turn the retrieval query into
        # a user-facing heading.
        sections: list[str] = []
        titles = {
            "issue": "## 研究关注与研究议题分布",
            "problem": "## 核心研究问题",
            "method": "## 主要方法路径",
            "finding": "## 研究发现与认识",
        }
        for group_type in ["issue", "problem", "method", "finding"]:
            sections.append(titles[group_type])
            synthesis = by_type.get(group_type)
            if synthesis is None:
                sections.append("候选文献中暂未形成可进入主要综述片段的知识单元组。")
            else:
                sections.append(synthesis.text)
            sections.append("")
        return "\n".join(sections).strip() + "\n"
