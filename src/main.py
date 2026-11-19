from __future__ import annotations

import argparse
from pathlib import Path

from src.orchestration.coordinator import Coordinator
from src.utils.env import load_project_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run abstract-level automated literature review prototype.")
    parser.add_argument("--query", required=True, help="Research topic, research question, or retrieval query.")
    parser.add_argument(
        "--case-id", "--topic-id", dest="case_id", default="T01",
        help="Topic identifier used as result/<model>/<topic>/ (default: T01).",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Model folder name; defaults to LLM_MODEL from the project environment.",
    )
    parser.add_argument(
        "--initial-top-k",
        type=int,
        default=40,
        help="Number of papers requested from the initial retrieval source before deduplication (default: 40).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Maximum number of deduplicated initial papers passed to downstream processing (default: 30).",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--graph-threshold", type=float, default=0.60)
    parser.add_argument(
        "--feedback-candidates-per-query", type=int, default=5,
        help="Maximum papers retrieved for each feedback gap query (default: 5).",
    )
    parser.add_argument(
        "--from-group-tables-case",
        default="",
        help="Reuse group_contribution_tables.json, papers.json, and contribution_records.json from an existing case, then rerun synthesis only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_project_env(project_root)
    coordinator = Coordinator(
        project_root=project_root,
        case_id=args.case_id,
        graph_threshold=args.graph_threshold,
        model_name=args.model_name or None,
        feedback_candidates_per_query=args.feedback_candidates_per_query,
    )
    if args.from_group_tables_case:
        output_dir = coordinator.run_from_group_tables(args.query, args.from_group_tables_case)
    else:
        output_dir = coordinator.run(
            args.query,
            initial_top_k=args.initial_top_k,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
