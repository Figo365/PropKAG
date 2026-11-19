from __future__ import annotations

import re
from typing import Any

from src.contribution.schemas import ContributionRecord, Paper
from src.contribution.unit_indexer import MAX_UNITS_PER_TYPE
from src.utils.llm_client import LLMClient


PROBLEM_PATTERNS = [
    r"\b(?:this (?:paper|study|work)|the paper|we|it)\s+(?:investigates?|addresses?|studies|focuses on|explores?|examines?)\b",
    r"(?:本文|本研究|该研究)(?:聚焦|探讨|研究|考察|解决|针对)",
]
METHOD_PATTERNS = [
    r"\b(?:this (?:paper|study|work)|the paper|we|it)\s+(?:proposes?|introduces?|develops?|presents?|uses?|applies|employs?|adopts?|builds?|constructs?|implements?|designs?)\b",
    r"\b(?:the|our)\s+(?:method|approach|framework|model|system|pipeline|algorithm)\s+(?:uses?|combines?|applies|employs?|adopts?|builds?|constructs?|implements?|integrates?)\b",
    r"(?:本文|本研究|该研究|我们)(?:提出|介绍|构建|设计|开发|采用|使用|应用|实现).{0,24}(?:方法|模型|框架|系统|算法|流程|策略)",
]
FINDING_PATTERNS = [
    r"\b(?:results?|findings?|experiments?|evaluation|analysis)\s+(?:show|shows|indicate|indicates|suggest|suggests|demonstrate|demonstrates|reveal|reveals)\b",
    r"\bwe\s+(?:find|found|observe|observed|show|demonstrate|report)\b",
    r"(?:结果|实验|评估|分析|研究发现)(?:表明|显示|证实|发现|指出)",
]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]


def clean_phrase(text: str, max_words: int = 28) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


class RuleBasedContributionExtractor:
    def extract(self, paper: Paper) -> ContributionRecord:
        sentences = split_sentences(paper.abstract)
        issues = [clean_phrase(paper.title)]

        problems = []
        methods = []
        findings = []
        for sentence in sentences:
            lower = sentence.lower()
            if self._matches(PROBLEM_PATTERNS, lower):
                problems.append(clean_phrase(sentence))
            if self._matches(METHOD_PATTERNS, lower):
                methods.append(clean_phrase(sentence))
            if self._matches(FINDING_PATTERNS, lower):
                findings.append(clean_phrase(sentence))

        evidence_text = {
            "title": paper.title,
            "abstract": paper.abstract,
            "keywords": "; ".join(paper.keywords),
            "issues": " | ".join(issues),
            "research_problems": " | ".join(problems),
            "research_methods": " | ".join(methods),
            "research_findings": " | ".join(findings),
        }
        return ContributionRecord(
            paper_id=paper.paper_id,
            issues=issues[:3],
            research_problems=problems[:MAX_UNITS_PER_TYPE],
            research_methods=methods[:MAX_UNITS_PER_TYPE],
            research_findings=findings[:MAX_UNITS_PER_TYPE],
            evidence_text=evidence_text,
            paper_summary=self._paper_summary(paper),
        )

    def _paper_summary(self, paper: Paper) -> str:
        sentences = split_sentences(paper.abstract)
        return clean_phrase(sentences[0] if sentences else paper.title, max_words=32)

    def _matches(self, patterns: list[str], text: str) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


class LLMContributionExtractor:
    def __init__(self, llm_client: LLMClient, prompt: str):
        self.llm_client = llm_client
        self.prompt = prompt
        self.last_mode = "llm"
        self.errors: list[dict[str, str]] = []

    def extract(self, paper: Paper, review_topic: str = "") -> ContributionRecord:
        payload = {
            "review_topic": review_topic,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "keywords": paper.keywords,
            "authors": paper.authors,
            "year": paper.year,
            "journal": paper.journal,
        }
        data = self.llm_client.chat_json(self.prompt, payload, temperature=0.0)
        return self._to_record(paper, data)

    def _to_record(self, paper: Paper, data: dict[str, Any]) -> ContributionRecord:
        paper_relevance = data.get("paper_relevance")
        if paper_relevance is False:
            return ContributionRecord(
                paper_id=paper.paper_id,
                issues=[],
                research_problems=[],
                research_methods=[],
                research_findings=[],
                evidence_text={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "keywords": "; ".join(paper.keywords),
                    "paper_relevance": "false",
                },
                paper_summary=self._paper_summary(data.get("paper_summary"), paper),
            )
        issues = self._string_list(data.get("issues"))[:MAX_UNITS_PER_TYPE]
        problems = self._string_list(data.get("research_problems"))[:MAX_UNITS_PER_TYPE]
        methods = self._string_list(data.get("research_methods"))[:MAX_UNITS_PER_TYPE]
        findings = self._string_list(data.get("research_findings"))[:MAX_UNITS_PER_TYPE]
        evidence = data.get("evidence_text") if isinstance(data.get("evidence_text"), dict) else {}
        evidence_text = {
            "title": paper.title,
            "abstract": paper.abstract,
            "keywords": "; ".join(paper.keywords),
            "issues": self._evidence_value(evidence.get("issues"), issues),
            "research_problems": self._evidence_value(evidence.get("research_problems"), problems),
            "research_methods": self._evidence_value(evidence.get("research_methods"), methods),
            "research_findings": self._evidence_value(evidence.get("research_findings"), findings),
            "paper_relevance": "true" if paper_relevance is True else "",
        }
        return ContributionRecord(
            paper_id=paper.paper_id,
            issues=issues,
            research_problems=problems,
            research_methods=methods,
            research_findings=findings,
            evidence_text=evidence_text,
            paper_summary=self._paper_summary(data.get("paper_summary"), paper),
        )

    def _paper_summary(self, value: Any, paper: Paper) -> str:
        background = clean_phrase(" ".join(str(value or "").split()), max_words=32)
        if background:
            return background
        sentences = split_sentences(paper.abstract)
        return clean_phrase(sentences[0] if sentences else paper.title, max_words=32)

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            values = [str(v).strip() for v in value if str(v).strip()]
        elif isinstance(value, str) and value.strip():
            values = [value.strip()]
        else:
            values = []
        deduplicated: list[str] = []
        seen: set[str] = set()
        for item in values:
            key = " ".join(item.split()).casefold()
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        return deduplicated

    def _evidence_value(self, value: Any, fallback: list[str]) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            return " | ".join(str(v).strip() for v in value if str(v).strip())
        return " | ".join(fallback)


class ContributionExtractor:
    def __init__(self, llm_client: LLMClient | None = None, prompt: str | None = None):
        self.rule_based = RuleBasedContributionExtractor()
        self.llm_based = LLMContributionExtractor(llm_client, prompt) if llm_client and llm_client.enabled and prompt else None
        self.last_mode = "rule_based"
        self.errors: list[dict[str, str]] = []

    def extract(self, paper: Paper, review_topic: str = "") -> ContributionRecord:
        if self.llm_based:
            try:
                record = self.llm_based.extract(paper, review_topic=review_topic)
                self.last_mode = "llm"
                return record
            except Exception as exc:
                self.errors.append({"paper_id": paper.paper_id, "error": str(exc)})
        self.last_mode = "rule_based"
        return self.rule_based.extract(paper)
