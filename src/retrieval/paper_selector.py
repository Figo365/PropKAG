from __future__ import annotations

from src.contribution.schemas import Paper
from src.utils.paper_identity import paper_identity_keys


class PaperSelector:
    def deduplicate(self, papers: list[Paper]) -> list[Paper]:
        seen: set[str] = set()
        selected: list[Paper] = []
        for paper in papers:
            keys = set(paper_identity_keys(paper))
            if not keys or keys & seen:
                continue
            seen.update(keys)
            selected.append(paper)
        return selected

    def filter_by_retrieval_score(self, papers: list[Paper], min_score: float = 0.0) -> list[Paper]:
        return [paper for paper in papers if paper.retrieval_score >= min_score]
