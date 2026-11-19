from __future__ import annotations

from pathlib import Path

from src.agents.base import BaseAgent
from src.contribution.schemas import Paper
from src.retrieval.paper_selector import PaperSelector
from src.retrieval.semantic_retriever import RetrievalError, SemanticRetriever
from src.utils.io import write_json


class RetrievalAgent(BaseAgent):
    name = "RetrievalAgent"

    def run(
        self,
        query: str,
        input_path: Path,
        initial_top_k: int,
        top_k: int,
        min_score: float,
    ) -> list[Paper]:
        task = self.state.start_task(self.name, [str(input_path), f"query:{query}"])
        retriever = SemanticRetriever(input_path)
        selector = PaperSelector()
        try:
            papers = retriever.search(query, top_k=initial_top_k)
        except RetrievalError as exc:
            response_output = write_json(
                self.output_dir / "retrieval_response.json",
                {
                    "retrieval_source": retriever.last_source,
                    "error": retriever.last_error or str(exc),
                    "response": retriever.last_response,
                },
            )
            output = write_json(self.output_dir / "papers.json", [])
            issues = [{"stage": "retrieval", "error": str(exc)}]
            self.state.finish_task(task, [response_output, output], status="failed", issues=issues)
            self.state.add_message(
                self.name,
                "Coordinator",
                "retrieval_failed",
                {"retrieval_source": retriever.last_source, "error": str(exc)},
                task.task_id,
            )
            self.state.flush()
            raise
        retrieved_count = len(papers)
        # Enforce the initial retrieval budget even if the upstream source
        # returns more records than requested; keep its raw count for auditing.
        papers = papers[:initial_top_k]
        initial_considered_count = len(papers)
        papers = selector.deduplicate(papers)
        deduplicated_count = len(papers)
        papers = papers[:top_k]
        papers = selector.filter_by_retrieval_score(papers, min_score)
        response_output = write_json(
            self.output_dir / "retrieval_response.json",
            {
                "retrieval_source": retriever.last_source,
                "error": retriever.last_error,
                "response": retriever.last_response,
                "initial_requested_count": initial_top_k,
                "retrieved_count": retrieved_count,
                "initial_considered_count": initial_considered_count,
                "deduplicated_count": deduplicated_count,
                "downstream_limit": top_k,
                "downstream_paper_count": len(papers),
            },
        )
        output = write_json(self.output_dir / "papers.json", papers)
        self.state.finish_task(task, [response_output, output])
        self.state.add_message(
            self.name,
            "ContributionExtractionAgent",
            "retrieve_completed",
            {
                "paper_count": len(papers),
                "retrieval_source": retriever.last_source,
                "error": retriever.last_error,
                "initial_requested_count": initial_top_k,
                "deduplicated_count": deduplicated_count,
                "downstream_limit": top_k,
            },
            task.task_id,
        )
        return papers
