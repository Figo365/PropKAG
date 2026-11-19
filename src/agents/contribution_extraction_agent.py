from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from src.agents.base import BaseAgent
from src.contribution.extractor import ContributionExtractor
from src.contribution.schemas import ContributionRecord, Paper
from src.contribution.unit_indexer import UnitIndexer
from src.utils.llm_client import LLMClient
from src.utils.io import write_json


class ContributionExtractionAgent(BaseAgent):
    name = "ContributionExtractionAgent"

    def run(self, papers: list[Paper], query: str = "", prompts: dict[str, str] | None = None) -> tuple[list[ContributionRecord], list]:
        task = self.state.start_task(self.name, ["papers.json", "prompts_used.json"])
        llm_client = LLMClient.from_env("LLM_BASIC")
        prompt = (prompts or {}).get("contribution_extraction")
        workers = self._worker_count("LLM_CONTRIBUTION_WORKERS", default=10)

        def extract_one(paper: Paper) -> tuple[ContributionRecord, str, list[dict[str, str]]]:
            # Each call gets isolated extractor state, so fallback errors and
            # mode tracking remain deterministic under concurrent execution.
            extractor = ContributionExtractor(llm_client=llm_client, prompt=prompt)
            record = extractor.extract(paper, review_topic=query)
            return record, extractor.last_mode, extractor.errors

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(papers)))) as executor:
            outcomes = list(executor.map(extract_one, papers))
        records = [record for record, _, _ in outcomes]
        modes = {mode for _, mode, _ in outcomes}
        errors = [error for _, _, item_errors in outcomes for error in item_errors]
        units = UnitIndexer().expand(records)
        llm_status = {
            "llm_profile": "LLM_BASIC",
            "llm_enabled": llm_client.enabled,
            "llm_model": llm_client.model,
            "extraction_mode": next(iter(modes)) if len(modes) == 1 else "mixed",
            "worker_count": min(workers, len(papers)),
            "fallback_errors": errors,
        }
        outputs = [
            write_json(self.output_dir / "contribution_records.json", records),
            write_json(self.output_dir / "knowledge_units.json", units),
            write_json(self.output_dir / "llm_status.json", llm_status),
        ]
        self.state.finish_task(task, outputs, issues=errors)
        self.state.add_message(
            self.name,
            "KnowledgeOrganizationAgent",
            "extraction_completed",
            {"record_count": len(records), "unit_count": len(units), **llm_status},
            task.task_id,
        )
        return records, units

    @staticmethod
    def _worker_count(variable: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(variable) or default))
        except ValueError:
            return default
