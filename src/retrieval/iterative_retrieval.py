from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import time

from src.contribution.schemas import GroupGap, Paper, RetrievalPlan
from src.retrieval.semantic_retriever import RetrievalError, SemanticRetriever
from src.utils.paper_identity import paper_identity_keys


class IterativeRetrieval:
    """Retrieve candidates with the query produced by group-gap diagnosis.

    The diagnosis prompt builds one query from both the target group's existing
    information and its most consequential gap. Gap-directed filtering,
    extraction, and fixed-group integration are handled by the following stages.
    """

    def __init__(
        self,
        input_path: Path,
        max_attempts: int = 3,
        retry_delay_seconds: float | None = None,
        candidates_per_query: int = 5,
    ):
        self.input_path = input_path
        self.max_attempts = max(1, max_attempts)
        self.candidates_per_query = max(1, candidates_per_query)
        self.worker_count = self._worker_count("LITERATURE_RETRIEVAL_WORKERS", default=4)
        self.batch_size = self._worker_count("LITERATURE_RETRIEVAL_BATCH_SIZE", default=self.worker_count)
        self.batch_delay_seconds = self._nonnegative_float("LITERATURE_RETRIEVAL_BATCH_DELAY_SECONDS", default=2.0)
        self.retry_delay_seconds = (
            max(0.0, retry_delay_seconds)
            if retry_delay_seconds is not None
            else self._nonnegative_float("LITERATURE_RETRIEVAL_RETRY_BASE_SECONDS", default=2.0)
        )
        self.retry_max_seconds = self._nonnegative_float("LITERATURE_RETRIEVAL_RETRY_MAX_SECONDS", default=30.0)

    def retrieve_candidates(
        self,
        gaps: list[GroupGap],
    ) -> tuple[list[RetrievalPlan], list[dict], list[dict]]:
        indexed_gaps = list(enumerate(sorted(gaps, key=self._gap_sort_key), start=1))

        # Retrieval state lives on SemanticRetriever, so every task owns a
        # separate client. Process bounded waves rather than immediately
        # queueing every gap against the upstream gateway.
        outcomes = []
        with ThreadPoolExecutor(max_workers=min(self.worker_count, max(1, len(indexed_gaps)))) as executor:
            for start in range(0, len(indexed_gaps), self.batch_size):
                batch = indexed_gaps[start : start + self.batch_size]
                outcomes.extend(executor.map(self._retrieve_gap, batch))
                if start + self.batch_size < len(indexed_gaps) and self.batch_delay_seconds:
                    time.sleep(self.batch_delay_seconds)

        plans: list[RetrievalPlan] = []
        candidates: list[dict] = []
        rejected: list[dict] = []
        for plan, gap_candidates, gap_rejected in outcomes:
            plans.append(plan)
            candidates.extend(gap_candidates)
            rejected.extend(gap_rejected)
        return plans, candidates, rejected

    def _retrieve_gap(self, indexed_gap: tuple[int, GroupGap]) -> tuple[RetrievalPlan, list[dict], list[dict]]:
        gap_index, gap = indexed_gap
        plan = RetrievalPlan(
            plan_id=f"PLAN-R{gap.round_id:02d}-{gap_index:02d}",
            round_id=gap.round_id,
            gap_id=gap.gap_id,
            group_id=gap.group_id,
            query=gap.retrieval_query,
        )
        retriever = SemanticRetriever(self.input_path)
        papers, attempt_log = self._search_with_retry(retriever, plan.query)
        candidates: list[dict] = []
        rejected: list[dict] = []
        if not papers:
            rejected.append(
                {
                    "gap_id": gap.gap_id,
                    "group_id": gap.group_id,
                    "title": "",
                    "reason": "retrieval_exhausted",
                    "attempt_count": len(attempt_log),
                    "attempts": attempt_log,
                }
            )
            return plan, candidates, rejected

        gap_seen: set[str] = set()
        for paper in papers:
            candidate = replace(paper, query_id=f"{plan.plan_id}-Q01")
            keys = set(paper_identity_keys(candidate))
            if not keys or keys & gap_seen:
                rejected.append({"gap_id": gap.gap_id, "title": candidate.title, "reason": "duplicate_retrieval_candidate"})
                continue
            gap_seen.update(keys)
            candidates.append({"gap_id": gap.gap_id, "group_id": gap.group_id, "paper": asdict(candidate)})
        return plan, candidates, rejected

    @staticmethod
    def _worker_count(variable: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(variable) or default))
        except ValueError:
            return default

    @staticmethod
    def _nonnegative_float(variable: str, default: float) -> float:
        try:
            return max(0.0, float(os.getenv(variable) or default))
        except ValueError:
            return default

    def _search_with_retry(self, retriever: SemanticRetriever, query: str) -> tuple[list[Paper], list[dict]]:
        attempts: list[dict] = []
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                papers = retriever.search(query, top_k=self.candidates_per_query)
                if papers:
                    papers = papers[:self.candidates_per_query]
                    attempts.append({"attempt": attempt, "status": "success", "result_count": len(papers), "error": ""})
                    return papers, attempts
                error = "Semantic retrieval returned no papers."
                attempts.append({"attempt": attempt, "status": "empty", "result_count": 0, "error": error})
            except RetrievalError as exc:
                last_error = retriever.last_error or str(exc)
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "result_count": 0,
                        "error": last_error,
                    }
                )
            if attempt < self.max_attempts and self.retry_delay_seconds:
                time.sleep(self._retry_delay(last_error, attempt))
        return [], attempts

    def _retry_delay(self, error: str, attempt: int) -> float:
        """Back off exponentially with jitter for transient gateway failures."""
        if "HTTP Error 502" in error:
            exponential = min(self.retry_max_seconds, self.retry_delay_seconds * (2 ** (attempt - 1)))
            return exponential + random.uniform(0.0, exponential * 0.25)
        return self.retry_delay_seconds * attempt

    def _gap_sort_key(self, gap: GroupGap) -> str:
        return gap.gap_id
