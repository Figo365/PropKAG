from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from dataclasses import asdict
from collections import Counter

from src.agents.contribution_extraction_agent import ContributionExtractionAgent
from src.agents.knowledge_organization_agent import KnowledgeOrganizationAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.contribution.schemas import (
    ContributionRecord,
    GroupContribution,
    GroupContributionTable,
    GroupContext,
    GroupContextUnit,
    GroupUnitRef,
    KnowledgeUnit,
    Paper,
    RoundMetrics,
    UnitGroup,
)
from src.generation.prompt_manager import PromptManager
from src.organization.group_gap_diagnoser import GroupGapDiagnoser
from src.organization.group_candidate_filter import GroupCandidateFilter
from src.organization.group_incremental_integrator import GroupIncrementalIntegrator
from src.organization.group_synthesis_input_builder import GroupContextBuilder
from src.organization.group_table_builder import GroupContributionTableBuilder
from src.orchestration.run_state import RunState
from src.retrieval.iterative_retrieval import IterativeRetrieval
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, read_json, write_json, write_text


GENERATED_OUTPUTS = [
    "data_manifest.json",
    "artifact_index.json",
    "papers.json",
    "retrieval_response.json",
    "contribution_records.json",
    "llm_status.json",
    "knowledge_units.json",
    "embedding_status.json",
    "similarity_matrix_issue.json",
    "similarity_matrix_problem.json",
    "similarity_matrix_method.json",
    "similarity_matrix_finding.json",
    "unit_relations.json",
    "detected_communities.json",
    "unit_groups.json",
    "unit_group_diagnostics.json",
    "group_naming_llm_status.json",
    "group_contribution_tables.json",
    "group_contexts.json",
    "review_fragments.json",
    "synthesis_llm_status.json",
    "type_syntheses.json",
    "type_integration_llm_status.json",
    "final_integration_llm_status.json",
    "final_review.md",
    "prompts_used.json",
    "agent_tasks.json",
    "agent_messages.json",
    "run_log.json",
    "iterative_retrieval_config.json",
    "iterative_retrieval_summary.json",
    "group_retrieval_attempts.json",
    "resume_source.json",
]

DATA_SCHEMA_VERSION = 7
FIXED_MAX_RETRIEVAL_ROUNDS = 1

# Reader-facing views link to the authoritative artifacts in the case root.
# This keeps audit files convenient to browse without making duplicate copies.
OUTPUT_VIEW_ARTIFACTS = {
    "intermediate_evidence": [
        "papers.json",
        "retrieval_response.json",
        "contribution_records.json",
        "knowledge_units.json",
        "unit_relations.json",
        "detected_communities.json",
        "unit_groups.json",
        "unit_group_diagnostics.json",
        "group_contribution_tables.json",
        "group_contexts.json",
        "review_fragments.json",
        "iterative_retrieval_config.json",
        "iterative_retrieval_summary.json",
        "group_retrieval_attempts.json",
    ],
    "final_deliverables": [
        "final_review.md",
    ],
}
OUTPUT_VIEW_PATTERNS = {
    "intermediate_evidence": [
        "gap_diagnoses_round_*.json",
        "retrieval_plans_round_*.json",
        "retrieval_candidates_round_*.json",
        "admitted_papers_round_*.json",
        "rejected_candidates_round_*.json",
        "group_snapshots_round_*.json",
        "group_integrations_round_*.json",
        "review_fragments_round_*.json",
        "group_review_updates_round_*.json",
        "feedback_revision_contexts_round_*.json",
        "round_metrics_round_*.json",
    ],
}

LEGACY_GENERATED_OUTPUTS = [
    "group_synthesis_inputs.json",
    "synthesis_context_cards.json",
    "unit_relations_candidate.json",
    "unit_relations_accepted.json",
    "weighted_graphs.json",
    "evidence_store.json",
    "final_verification_summary.json",
    "final_review_validation_report.json",
    "final_review_validation_report.md",
    "final_review_before_verification.md",
    "final_review_unrepaired.md",
]
LEGACY_GENERATED_PATTERNS = [
    "claim_evidence_table_round_*.json",
    "verification_report_round_*.json",
    "verification_llm_status_round_*.json",
    "final_review_repaired_round_*.md",
    "synthesis_repair_llm_status_round_*.json",
]

class Coordinator:
    def __init__(
        self,
        project_root: Path,
        case_id: str,
        graph_threshold: float = 0.60,
        model_name: str | None = None,
        feedback_candidates_per_query: int = 5,
    ):
        if type(feedback_candidates_per_query) is not int or feedback_candidates_per_query < 1:
            raise ValueError("feedback_candidates_per_query must be a positive integer.")
        self.project_root = project_root
        load_project_env(self.project_root)
        self.model_name = str(model_name or os.getenv("LLM_MODEL") or "unconfigured-model").strip()
        self.model_folder = self._safe_model_folder(self.model_name)
        self.output_root = ensure_dir(project_root / "result" / self.model_folder)
        self.case_id = self._validate_case_id(case_id, "case_id")
        self.topic_id = self.case_id
        self.graph_threshold = graph_threshold
        self.feedback_candidates_per_query = feedback_candidates_per_query
        self.max_retrieval_rounds = FIXED_MAX_RETRIEVAL_ROUNDS
        self.result_dir = ensure_dir(self._case_directory(self.case_id, "case_id"))
        self.output_dir = ensure_dir(self.result_dir / "audit_artifacts")
        self.final_review_path = self.result_dir / f"{self.topic_id}_final_review.md"
        self.state = RunState(self.output_dir)

    def _validate_case_id(self, value: str, argument_name: str) -> str:
        case_id = str(value or "").strip()
        candidate = Path(case_id)
        if (
            not case_id
            or case_id in {".", ".."}
            or candidate.is_absolute()
            or "/" in case_id
            or "\\" in case_id
            or len(candidate.parts) != 1
        ):
            raise ValueError(
                f"{argument_name} must be a single output directory name, without path separators."
            )
        return case_id

    @staticmethod
    def _safe_model_folder(value: str) -> str:
        folder = re.sub(r"[^\w.-]+", "_", str(value or "").strip(), flags=re.UNICODE)
        folder = folder.strip("._-")
        return folder or "unconfigured-model"

    def _case_directory(self, case_id: str, argument_name: str) -> Path:
        safe_case_id = self._validate_case_id(case_id, argument_name)
        candidate = self.output_root / safe_case_id
        try:
            resolved_root = self.output_root.resolve()
            resolved_candidate = candidate.resolve()
        except OSError as exc:
            raise ValueError(f"Unable to resolve {argument_name}: {safe_case_id}") from exc
        if resolved_candidate.parent != resolved_root:
            raise ValueError(f"{argument_name} must resolve directly inside {resolved_root}.")
        return candidate

    def run(
        self,
        query: str,
        initial_top_k: int = 40,
        top_k: int = 30,
        min_score: float = 0.0,
    ) -> Path:
        self._clean_known_outputs()
        self._write_data_manifest(initial_top_k=initial_top_k, top_k=top_k, min_score=min_score)
        prompts = self._load_prompts()
        write_json(self.output_dir / "prompts_used.json", prompts)
        self.state.log("prompts_loaded", {"prompt_count": len(prompts), "prompt_names": sorted(prompts)})

        input_path = self.project_root / "data" / "input" / "papers.json"
        retrieval = RetrievalAgent(self.output_dir, self.state)
        extraction = ContributionExtractionAgent(self.output_dir, self.state)
        organization = KnowledgeOrganizationAgent(self.output_dir, self.state)
        synthesis = SynthesisAgent(self.output_dir, self.state)

        papers = retrieval.run(query, input_path, initial_top_k, top_k, min_score)
        records, units = extraction.run(papers, query=query, prompts=prompts)
        groups, tables, group_contexts = organization.run(
            units,
            records,
            self.graph_threshold,
            review_topic=query,
            prompts=prompts,
        )
        fragments = synthesis.generate_fragments(query, tables, prompts=prompts, group_contexts=group_contexts)
        write_json(self.output_dir / "review_fragments_round_00.json", fragments)
        papers, records, groups, tables, group_contexts, fragments = self._run_iterative_retrieval(
            query=query,
            input_path=input_path,
            papers=papers,
            records=records,
            units=units,
            groups=groups,
            tables=tables,
            group_contexts=group_contexts,
            fragments=fragments,
            synthesis=synthesis,
            prompts=prompts,
        )
        type_syntheses = synthesis.generate_type_syntheses(query, fragments, prompts=prompts)
        final_text = synthesis.integrate(query, type_syntheses, prompts=prompts)
        write_text(self.final_review_path, final_text)
        self.state.flush()
        self._write_output_views()
        return self.result_dir

    def run_from_group_tables(self, query: str, source_case_id: str) -> Path:
        source_case_id = self._validate_case_id(source_case_id, "source_case_id")
        source_dir = self._case_directory(source_case_id, "source_case_id")
        if not source_dir.exists():
            raise FileNotFoundError(f"Source case does not exist: {source_dir}")
        if source_dir.resolve() == self.result_dir.resolve():
            raise ValueError("Use a different --case-id when rerunning from existing group_contribution_tables.json.")
        self._validate_source_schema(source_dir)
        source_artifact_dir = source_dir / "audit_artifacts"
        if not source_artifact_dir.is_dir():
            raise FileNotFoundError(f"Source audit artifacts do not exist: {source_artifact_dir}")

        self._clean_known_outputs()
        self._write_data_manifest(resume_source_case_id=source_case_id)
        prompts = self._load_prompts()
        write_json(self.output_dir / "prompts_used.json", prompts)
        self.state.log(
            "prompts_loaded",
            {
                "prompt_count": len(prompts),
                "prompt_names": sorted(prompts),
                "resume_mode": "from_group_tables",
                "source_case_id": source_case_id,
            },
        )

        papers = self._load_dataclass_list(source_artifact_dir / "papers.json", Paper)
        records = self._load_dataclass_list(source_artifact_dir / "contribution_records.json", ContributionRecord)
        source_tables = self._load_group_tables(source_artifact_dir / "group_contribution_tables.json")
        tables, group_contexts, context_mode = self._load_or_rebuild_group_context(
            source_artifact_dir, source_tables, records, query
        )
        write_json(
            self.output_dir / "resume_source.json",
            {
                "mode": "from_group_tables",
                "source_case_id": source_case_id,
                "source_dir": str(source_dir),
                "paper_count": len(papers),
                "record_count": len(records),
                "table_count": len(tables),
                "group_context_count": len(group_contexts),
                "group_context_mode": context_mode,
                "current_review_topic": query,
                "review_topic_overridden": True,
            },
        )
        write_json(self.output_dir / "papers.json", papers)
        write_json(self.output_dir / "contribution_records.json", records)
        write_json(self.output_dir / "group_contribution_tables.json", tables)
        write_json(self.output_dir / "group_contexts.json", group_contexts)
        # A synthesis-only resume must retain the source grouping artifacts so
        # its output remains auditable under the current data manifest.
        for artifact_name in ("knowledge_units.json", "unit_groups.json"):
            artifact = read_json(source_artifact_dir / artifact_name, default=None)
            if isinstance(artifact, list):
                write_json(self.output_dir / artifact_name, artifact)

        synthesis = SynthesisAgent(self.output_dir, self.state)
        fragments = synthesis.generate_fragments(query, tables, prompts=prompts, group_contexts=group_contexts)
        type_syntheses = synthesis.generate_type_syntheses(query, fragments, prompts=prompts)
        final_text = synthesis.integrate(query, type_syntheses, prompts=prompts)
        write_text(self.final_review_path, final_text)
        self.state.flush()
        self._write_output_views()
        return self.result_dir

    def _load_prompts(self) -> dict[str, str]:
        names = [
            "contribution_extraction",
            "unit_group_naming",
            "group_iterative_gap_diagnosis",
            "group_gap_filter_and_extract",
            "group_feedback_revision",
            "issue_group_synthesis",
            "problem_group_synthesis",
            "method_group_synthesis",
            "finding_group_synthesis",
            "other_group_synthesis",
            "issue_type_integration",
            "problem_type_integration",
            "method_type_integration",
            "finding_type_integration",
        ]
        prompts = PromptManager(self.project_root / "config" / "prompts").load_many(names)
        prompts["group_iterative_gap_diagnosis"] = prompts["group_iterative_gap_diagnosis"].replace(
            "{{feedback_candidates_per_query}}", str(getattr(self, "feedback_candidates_per_query", 5))
        )
        return prompts

    def _load_dataclass_list(self, path: Path, cls):
        rows = read_json(path)
        if rows is None:
            raise FileNotFoundError(f"Required resume input is missing: {path}")
        if not isinstance(rows, list):
            raise ValueError(f"Expected list in {path}")
        fields = cls.__dataclass_fields__
        return [cls(**{key: value for key, value in row.items() if key in fields}) for row in rows]

    def _load_group_tables(self, path: Path) -> list[GroupContributionTable]:
        rows = read_json(path)
        if rows is None:
            raise FileNotFoundError(f"Required resume input is missing: {path}")
        if not isinstance(rows, list):
            raise ValueError(f"Expected list in {path}")
        tables: list[GroupContributionTable] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows in {path}")
            contributions: list[GroupContribution] = []
            for value in row.get("contributions", []):
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid group contribution in {path}")
                raw_units = value.get("units") or value.get("target_units") or []
                units = [
                    GroupUnitRef(
                        unit_id=str(unit.get("unit_id") or ""),
                        unit_type=str(unit.get("unit_type") or row.get("group_type") or ""),
                        text=str(unit.get("text") or ""),
                    )
                    for unit in raw_units
                    if isinstance(unit, dict) and str(unit.get("text") or "").strip()
                ]
                contributions.append(
                    GroupContribution(
                        paper_id=str(value.get("paper_id") or ""),
                        paper_summary=str(value.get("paper_summary") or ""),
                        units=units,
                    )
                )
            tables.append(
                GroupContributionTable(
                    group_id=str(row.get("group_id") or ""),
                    group_type=str(row.get("group_type") or ""),
                    group_label=str(row.get("group_label") or ""),
                    group_summary=str(row.get("group_summary") or ""),
                    group_quality_flag=str(row.get("group_quality_flag") or ""),
                    contributions=contributions,
                    group_role=str(row.get("group_role") or "semantic"),
                )
            )
        return tables

    def _load_group_contexts(self, source_dir: Path, tables: list[GroupContributionTable], review_topic: str = "") -> list[GroupContext]:
        rows = read_json(source_dir / "group_contexts.json")
        if isinstance(rows, list):
            fields = GroupContext.__dataclass_fields__
            contexts = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object rows in {source_dir / 'group_contexts.json'}")
                rows_units = row.get("units") if isinstance(row.get("units"), list) else []
                units = [
                    GroupContextUnit(**{key: value for key, value in value.items() if key in GroupContextUnit.__dataclass_fields__})
                    for value in rows_units
                    if isinstance(value, dict)
                ]
                normalized = {**row, "review_topic": review_topic, "units": units}
                required = ["group_id", "group_type", "group_label", "group_summary", "group_quality_flag", "size"]
                if any(normalized.get(key) is None for key in required):
                    raise ValueError(f"Incomplete group metadata in {source_dir / 'group_contexts.json'}")
                contexts.append(GroupContext(**{key: value for key, value in normalized.items() if key in fields}))
            return contexts

        # Migrate the former duplicated GroupSynthesisInput projection once at
        # load time; it is never written by the new pipeline.
        legacy_rows = read_json(source_dir / "group_synthesis_inputs.json")
        if isinstance(legacy_rows, list):
            tables_by_id = {table.group_id: table for table in tables}
            builder = GroupContextBuilder()
            contexts = []
            for row in legacy_rows:
                if not isinstance(row, dict):
                    continue
                group = row.get("group") if isinstance(row.get("group"), dict) else {}
                units = []
                for paper in row.get("representative_papers", []):
                    if not isinstance(paper, dict):
                        continue
                    paper_id = str(paper.get("paper_id") or "")
                    background = str(paper.get("paper_summary") or paper_id)
                    for unit in paper.get("target_units", []):
                        if isinstance(unit, dict) and str(unit.get("text") or "").strip():
                            units.append(
                                GroupContextUnit(
                                    unit_id=str(unit.get("unit_id") or ""),
                                    paper_id=paper_id,
                                    text=str(unit["text"]),
                                    background=background,
                                )
                            )
                context = GroupContext(
                    group_id=str(row.get("group_id") or group.get("group_id") or ""),
                    group_type=str(row.get("group_type") or group.get("group_type") or ""),
                    group_label=str(row.get("group_label") or group.get("group_label") or ""),
                    group_summary=str(group.get("group_summary") or ""),
                    group_quality_flag=str(row.get("group_quality_flag") or group.get("quality_flag") or ""),
                    size=len(units),
                    review_topic=review_topic,
                    units=units,
                    structure_scope="legacy_migrated_context",
                    initial_size=len(units),
                )
                if not context.units and context.group_id in tables_by_id:
                    context = builder.build_from_table(tables_by_id[context.group_id], review_topic=review_topic)
                contexts.append(context)
            return contexts
        builder = GroupContextBuilder()
        return [builder.build_from_table(table, review_topic=review_topic) for table in tables]

    def _load_or_rebuild_group_context(
        self,
        source_dir: Path,
        source_tables: list[GroupContributionTable],
        records: list[ContributionRecord],
        review_topic: str,
    ) -> tuple[list[GroupContributionTable], list[GroupContext], str]:
        group_rows = read_json(source_dir / "unit_groups.json")
        unit_rows = read_json(source_dir / "knowledge_units.json")
        if isinstance(group_rows, list) and isinstance(unit_rows, list):
            group_fields = UnitGroup.__dataclass_fields__
            unit_fields = KnowledgeUnit.__dataclass_fields__
            groups = [UnitGroup(**{key: value for key, value in row.items() if key in group_fields}) for row in group_rows]
            units = [KnowledgeUnit(**{key: value for key, value in row.items() if key in unit_fields}) for row in unit_rows]
            records_by_id = {record.paper_id: record for record in records}
            units_by_id = {unit.unit_id: unit for unit in units}
            context_complete = all(
                all(unit_id in units_by_id for unit_id in group.units)
                and all(paper_id in records_by_id for paper_id in group.representative_papers)
                for group in groups
            )
            if not context_complete:
                return source_tables, self._load_group_contexts(source_dir, source_tables, review_topic), "legacy_unscoped_context"
            table_builder = GroupContributionTableBuilder()
            context_builder = GroupContextBuilder()
            tables = [table_builder.build(group, records_by_id, units_by_id) for group in groups]
            contexts = [context_builder.build(review_topic, group, records_by_id, units_by_id) for group in groups]
            return tables, contexts, "rebuilt_strict_group_units"
        return source_tables, self._load_group_contexts(source_dir, source_tables, review_topic), "legacy_unscoped_context"

    def _clean_known_outputs(self) -> None:
        for path in (self.result_dir / "data_manifest.json", self.final_review_path):
            if path.is_file() or path.is_symlink():
                path.unlink()
        self._clean_output_views()
        names = [*GENERATED_OUTPUTS, *LEGACY_GENERATED_OUTPUTS]
        for name in names:
            path = self.output_dir / name
            if path.exists() and path.is_file():
                path.unlink()
        patterns = [
            pattern for patterns in OUTPUT_VIEW_PATTERNS.values() for pattern in patterns
        ] + LEGACY_GENERATED_PATTERNS
        for pattern in patterns:
            for path in self.output_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    def _clean_output_views(self) -> None:
        """Remove only generated links from the reader-facing output views."""
        for directory in OUTPUT_VIEW_ARTIFACTS:
            view_dir = self.output_dir / directory
            if not view_dir.exists() or not view_dir.is_dir():
                continue
            generated_names = set(OUTPUT_VIEW_ARTIFACTS[directory])
            generated_names.update(LEGACY_GENERATED_OUTPUTS)
            patterns = [*OUTPUT_VIEW_PATTERNS.get(directory, []), *LEGACY_GENERATED_PATTERNS]
            for pattern in patterns:
                generated_names.update(path.name for path in self.output_dir.glob(pattern))
                generated_names.update(path.name for path in view_dir.glob(pattern))
            for name in generated_names:
                path = view_dir / name
                if path.is_file() or path.is_symlink():
                    path.unlink()

    def _write_output_views(self) -> None:
        """Group audit artifacts and final deliverables without duplicating files."""
        view_index: dict[str, list[dict[str, str]]] = {}
        for directory, artifact_names in OUTPUT_VIEW_ARTIFACTS.items():
            view_dir = ensure_dir(self.output_dir / directory)
            linked: list[dict[str, str]] = []
            names = [*artifact_names]
            for pattern in OUTPUT_VIEW_PATTERNS.get(directory, []):
                names.extend(path.name for path in sorted(self.output_dir.glob(pattern)))
            for name in dict.fromkeys(names):
                source = self.output_dir / name
                if not source.is_file():
                    continue
                destination = view_dir / name
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                try:
                    os.link(source, destination)
                    link_mode = "hardlink"
                except OSError:
                    destination.symlink_to(os.path.relpath(source, view_dir))
                    link_mode = "symlink"
                linked.append({"artifact": name, "link_mode": link_mode})
            view_index[directory] = linked
        write_json(
            self.output_dir / "artifact_index.json",
            {
                "storage_policy": "Case-root files are authoritative; reader-facing folders are linked views without data copies.",
                "views": view_index,
            },
        )

    def _write_data_manifest(
        self, resume_source_case_id: str = "", *,
        initial_top_k: int | None = None, top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        # Hash the executed source and prompts without recording credentials.
        files = [
            *sorted((self.project_root / "src").rglob("*.py")),
            *sorted((self.project_root / "config" / "prompts").glob("*.txt")),
            *sorted((self.project_root / "config").glob("experiment*.json")),
        ]
        runner_path = self.project_root / "run_experiment.py"
        if runner_path.is_file():
            files.append(runner_path)
        file_hashes = {
            path.relative_to(self.project_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if "__pycache__" not in path.parts
        }
        try:
            max_gaps = min(3, max(1, int(os.getenv("MAX_GAP_DIRECTIONS_PER_GROUP") or 3)))
        except ValueError:
            max_gaps = 3
        source_settings = None
        if resume_source_case_id:
            source_manifest = read_json(
                self._case_directory(resume_source_case_id, "source_case_id") / "data_manifest.json"
            )
            source_settings = source_manifest["execution_settings"]
        write_json(
            self.result_dir / "data_manifest.json",
            {
                "schema_version": DATA_SCHEMA_VERSION,
                "generator_model": self.model_name,
                "model_folder": self.model_folder,
                "topic_id": self.topic_id,
                "execution_mode": "synthesis_only" if resume_source_case_id else "full_pipeline",
                "execution_settings": {
                    "initial_top_k": initial_top_k,
                    "top_k": top_k,
                    "min_score": min_score,
                    "graph_threshold": self.graph_threshold,
                    "graph_edge_rule": "threshold",
                    "leiden_resolution": (
                        source_settings["leiden_resolution"] if source_settings
                        else max(0.01, float(os.getenv("GROUP_LEIDEN_RESOLUTION", "1.0")))
                    ),
                    "leiden_seed": (
                        source_settings["leiden_seed"] if source_settings
                        else int(os.getenv("GROUP_LEIDEN_SEED", "42"))
                    ),
                    "feedback_rounds": 0 if resume_source_case_id else self.max_retrieval_rounds,
                    "max_gap_directions_per_group": max_gaps,
                    "feedback_candidates_per_query": self.feedback_candidates_per_query,
                },
                "source_and_prompt_sha256": file_hashes,
                "primary_artifacts": [
                    "audit_artifacts/papers.json",
                    "audit_artifacts/contribution_records.json",
                    "audit_artifacts/knowledge_units.json",
                    "audit_artifacts/unit_groups.json",
                    "audit_artifacts/unit_relations.json",
                ],
                "derived_artifacts": [
                    "audit_artifacts/group_contribution_tables.json",
                    "audit_artifacts/group_contexts.json",
                    "audit_artifacts/similarity_matrix_<type>.json",
                    f"{self.topic_id}_final_review.md",
                ],
                "storage_policy": "Primary artifacts are authoritative; derived artifacts are rebuildable projections.",
                "reader_facing_views": {
                    "intermediate_evidence": "Intermediate retrieval, organization, and feedback revision artifacts.",
                    "final_deliverables": "Final review.",
                },
                "resume_source_case_id": resume_source_case_id or None,
                "resume_source_execution_settings": source_settings,
            },
        )

    def _validate_source_schema(self, source_dir: Path) -> None:
        manifest = read_json(source_dir / "data_manifest.json", default=None)
        version = manifest.get("schema_version") if isinstance(manifest, dict) else None
        if type(version) is not int:
            raise ValueError(f"Invalid data_manifest.json in resume source: {source_dir}")
        if version > DATA_SCHEMA_VERSION:
            raise ValueError(
                f"Resume source schema v{version} is newer than supported v{DATA_SCHEMA_VERSION}: {source_dir}"
            )
        settings = manifest.get("execution_settings")
        if not isinstance(settings, dict) or settings.get("graph_edge_rule") != "threshold":
            raise ValueError("Resume requires a recorded threshold-only source run; rebuild the source grouping first.")
        threshold = settings.get("graph_threshold")
        if (
            type(threshold) not in (int, float)
            or not 0 <= threshold <= 1
            or threshold != self.graph_threshold
        ):
            raise ValueError("Resume source graph threshold must match the current graph threshold.")
        candidates = settings.get("feedback_candidates_per_query")
        if type(candidates) is not int or candidates != self.feedback_candidates_per_query:
            raise ValueError("Resume source feedback budget must match the current feedback budget.")
        resolution = settings.get("leiden_resolution")
        seed = settings.get("leiden_seed")
        if (
            type(resolution) not in (int, float)
            or not 0 < resolution < float("inf")
            or type(seed) is not int
            or seed < 1
        ):
            raise ValueError("Resume requires recorded Leiden resolution and seed settings.")

    def _run_iterative_retrieval(
        self,
        query: str,
        input_path: Path,
        papers: list[Paper],
        records: list[ContributionRecord],
        units,
        groups,
        tables: list[GroupContributionTable],
        group_contexts: list[GroupContext],
        fragments,
        synthesis: SynthesisAgent,
        prompts: dict[str, str],
    ):
        excluded_other_group_ids = sorted(
            group.group_id for group in groups if group.group_role == "other"
        )
        diagnoser = GroupGapDiagnoser(prompts=prompts)
        retriever = IterativeRetrieval(input_path, candidates_per_query=self.feedback_candidates_per_query)
        filterer = GroupCandidateFilter(prompts=prompts)
        config = {
            "enabled": True,
            "max_retrieval_rounds": self.max_retrieval_rounds,
            "gap_diagnosis_mode": "single_pass_multiple_directions",
            "max_gap_directions_per_group": diagnoser.max_directions,
            "candidates_per_query": retriever.candidates_per_query,
            "candidate_admission_policy": "evaluate_every_candidate_with_llm",
            "max_attempts_per_query": retriever.max_attempts,
            "gap_diagnosis_source": "group_context_and_group_review",
            "no_gap_groups_are_final": True,
            "excluded_group_roles": ["other"],
            "excluded_other_group_ids": excluded_other_group_ids,
        }
        config.update(
            {
                "max_gap_directions_per_group": diagnoser.max_directions,
                "gap_diagnosis_workers": diagnoser.worker_count,
                "retrieval_workers": retriever.worker_count,
                "retrieval_batch_size": retriever.batch_size,
                "retrieval_batch_delay_seconds": retriever.batch_delay_seconds,
                "retrieval_retry_base_seconds": retriever.retry_delay_seconds,
                "retrieval_retry_max_seconds": retriever.retry_max_seconds,
                "candidate_filter_workers": filterer.worker_count,
                "changed_group_review_workers": synthesis._worker_count("LLM_SYNTHESIS_WORKERS", default=10),
                "integration_merge_mode": "deterministic_input_order",
            }
        )
        write_json(self.output_dir / "iterative_retrieval_config.json", config)
        integrator = GroupIncrementalIntegrator()
        summary = {
            "config": config,
            "rounds": [],
            "stop_reason": "max_retrieval_rounds_reached",
            "excluded_other_group_ids": excluded_other_group_ids,
        }
        attempts_by_group: dict[str, list[dict]] = {}
        exhausted_group_ids: set[str] = set()
        closed_no_gap_group_ids: set[str] = set()

        for round_id in range(1, self.max_retrieval_rounds + 1):
            paper_count_before = len(papers)
            group_count_before = len(groups)
            if not groups:
                summary["stop_reason"] = "no_knowledge_unit_groups"
                break
            eligible_groups = [
                group
                for group in groups
                if group.group_role != "other"
                and group.group_id not in exhausted_group_ids
                and group.group_id not in closed_no_gap_group_ids
            ]
            eligible_group_ids = {group.group_id for group in eligible_groups}
            eligible_reviews = [item for item in fragments if item.group_id in eligible_group_ids]
            if not eligible_groups:
                semantic_groups = [group for group in groups if group.group_role != "other"]
                summary["stop_reason"] = (
                    "no_gap_eligible_groups" if not semantic_groups else "all_groups_closed_or_exhausted"
                )
                break
            gaps, diagnosis_log = diagnoser.diagnose(
                query,
                eligible_groups,
                eligible_reviews,
                round_id,
                attempt_history=attempts_by_group,
                group_contexts=[item for item in group_contexts if item.group_id in eligible_group_ids],
            )
            closed_no_gap_group_ids.update(self._no_gap_group_ids(diagnosis_log))
            summary["closed_no_gap_group_ids"] = sorted(closed_no_gap_group_ids)
            write_json(self.output_dir / f"gap_diagnoses_round_{round_id:02d}.json", {"gaps": gaps, "diagnosis_log": diagnosis_log})
            if not gaps:
                logged_groups = list(diagnosis_log.get("groups", []))
                failed_groups = [item for item in logged_groups if item.get("diagnosis_status") == "diagnosis_failed"]
                if logged_groups and len(failed_groups) == len(logged_groups):
                    diagnosis_stop_reason = "gap_diagnosis_failed"
                elif failed_groups:
                    diagnosis_stop_reason = "gap_diagnosis_incomplete"
                elif any(item.get("diagnosis_status") == "query_rejected" for item in logged_groups):
                    diagnosis_stop_reason = "no_new_retrieval_query"
                else:
                    diagnosis_stop_reason = "no_open_retrieval_gaps"
                round_summary = self._round_metrics(
                    round_id=round_id,
                    paper_count_before=paper_count_before,
                    paper_count_after=len(papers),
                    group_count_before=group_count_before,
                    group_count_after=len(groups),
                    stop_reason=diagnosis_stop_reason,
                )
                write_json(self.output_dir / f"round_metrics_round_{round_id:02d}.json", round_summary)
                summary["rounds"].append(round_summary)
                summary["stop_reason"] = diagnosis_stop_reason
                break

            plans, candidates, retrieval_rejected = retriever.retrieve_candidates(gaps)
            newly_exhausted = {
                str(item.get("group_id"))
                for item in retrieval_rejected
                if item.get("reason") == "retrieval_exhausted" and item.get("group_id")
            }
            exhausted_group_ids.update(newly_exhausted)
            summary["exhausted_group_ids"] = sorted(exhausted_group_ids)
            accepted, filter_rejected = filterer.filter_and_extract(
                candidates=candidates,
                gaps=gaps,
                group_contexts=group_contexts,
                existing_papers=papers,
            )
            rejected = [*retrieval_rejected, *filter_rejected]
            write_json(self.output_dir / f"retrieval_plans_round_{round_id:02d}.json", plans)
            write_json(self.output_dir / f"retrieval_candidates_round_{round_id:02d}.json", candidates)
            write_json(self.output_dir / f"rejected_candidates_round_{round_id:02d}.json", rejected)

            if not accepted:
                write_json(self.output_dir / f"admitted_papers_round_{round_id:02d}.json", [])
                write_json(self.output_dir / f"group_integrations_round_{round_id:02d}.json", [])
                round_summary = self._round_metrics(
                    round_id=round_id,
                    open_gap_count=len(gaps),
                    planned_gap_count=len(plans),
                    candidate_count=len(candidates),
                    paper_count_before=paper_count_before,
                    paper_count_after=len(papers),
                    group_count_before=group_count_before,
                    group_count_after=len(groups),
                    round_outcome="no_papers_admitted",
                )
                self._append_group_retrieval_attempts(
                    attempts_by_group, gaps, candidates, accepted, rejected, [], round_id
                )
                write_json(self.output_dir / "group_retrieval_attempts.json", attempts_by_group)
                write_json(self.output_dir / f"round_metrics_round_{round_id:02d}.json", round_summary)
                summary["rounds"].append(round_summary)
                self.state.log("iterative_retrieval_round_completed", round_summary)
                continue

            groups_before = [asdict(group) for group in groups]
            fragments_before = [asdict(fragment) for fragment in fragments]
            papers, records, units, groups, tables, group_contexts, integrations = integrator.integrate(
                accepted=accepted,
                papers=papers,
                records=records,
                units=units,
                groups=groups,
                tables=tables,
                group_contexts=group_contexts,
                review_topic=query,
                round_id=round_id,
            )
            admitted = [paper for paper in papers if paper.retrieval_round == round_id and paper.source == "iterative_retrieval"]
            write_json(self.output_dir / f"admitted_papers_round_{round_id:02d}.json", admitted)
            write_json(self.output_dir / f"group_integrations_round_{round_id:02d}.json", integrations)
            write_json(self.output_dir / "papers.json", papers)
            write_json(self.output_dir / "contribution_records.json", records)
            write_json(self.output_dir / "knowledge_units.json", units)
            write_json(self.output_dir / "unit_groups.json", groups)
            write_json(self.output_dir / "group_contribution_tables.json", tables)
            write_json(self.output_dir / "group_contexts.json", group_contexts)
            write_json(self.output_dir / f"group_snapshots_round_{round_id:02d}.json", {"before": groups_before, "after": groups})
            changed_group_ids = {item.target_group_id for item in integrations}
            current_drafts = {fragment.group_id: fragment for fragment in fragments}
            papers_by_id = {paper.paper_id: paper for paper in papers}
            revision_contexts = {}
            for group_id in sorted(changed_group_ids):
                source_ids = sorted({
                    item.paper_id for item in integrations if item.target_group_id == group_id
                })
                revision_contexts[group_id] = {
                    "current_draft": asdict(current_drafts[group_id]),
                    "information_gaps": [asdict(gap) for gap in gaps if gap.group_id == group_id],
                    "supplementary_papers": [asdict(papers_by_id[paper_id]) for paper_id in source_ids],
                }
            write_json(self.output_dir / f"feedback_revision_contexts_round_{round_id:02d}.json", revision_contexts)
            fragments = synthesis.regenerate_group_fragments(
                query,
                tables,
                fragments,
                changed_group_ids,
                prompts=prompts,
                group_contexts=group_contexts,
                revision_contexts_by_group=revision_contexts,
            )
            write_json(self.output_dir / f"review_fragments_round_{round_id:02d}.json", fragments)
            write_json(
                self.output_dir / f"group_review_updates_round_{round_id:02d}.json",
                {
                    "round_id": round_id,
                    "updated_group_ids": sorted(changed_group_ids),
                    "before": fragments_before,
                    "after": [asdict(fragment) for fragment in fragments],
                },
            )
            round_outcome = "papers_integrated" if integrations else "no_units_integrated"
            round_summary = self._round_metrics(
                round_id=round_id,
                open_gap_count=len(gaps),
                planned_gap_count=len(plans),
                candidate_count=len(candidates),
                accepted_candidate_count=len(accepted),
                new_paper_count=len(papers) - paper_count_before,
                integrated_unit_count=len(integrations),
                paper_count_before=paper_count_before,
                paper_count_after=len(papers),
                group_count_before=group_count_before,
                group_count_after=len(groups),
                round_outcome=round_outcome,
            )
            self._append_group_retrieval_attempts(
                attempts_by_group, gaps, candidates, accepted, rejected, integrations, round_id
            )
            write_json(self.output_dir / "group_retrieval_attempts.json", attempts_by_group)
            write_json(self.output_dir / f"round_metrics_round_{round_id:02d}.json", round_summary)
            summary["rounds"].append(round_summary)
            self.state.log("iterative_retrieval_round_completed", round_summary)

        if summary["stop_reason"] == "max_retrieval_rounds_reached":
            summary["stop_reason"] = "single_pass_completed"
        write_json(self.output_dir / "iterative_retrieval_summary.json", summary)
        return papers, records, groups, tables, group_contexts, fragments

    def _no_gap_group_ids(self, diagnosis_log: dict) -> set[str]:
        return {
            str(item.get("group_id"))
            for item in diagnosis_log.get("groups", [])
            if item.get("diagnosis_status") == "no_gap" and item.get("group_id")
        }

    def _round_metrics(
        self,
        round_id: int,
        paper_count_before: int,
        paper_count_after: int,
        group_count_before: int,
        group_count_after: int,
        open_gap_count: int = 0,
        planned_gap_count: int = 0,
        candidate_count: int = 0,
        accepted_candidate_count: int = 0,
        new_paper_count: int = 0,
        integrated_unit_count: int = 0,
        round_outcome: str = "",
        stop_reason: str = "",
    ) -> dict:
        return asdict(
            RoundMetrics(
                round_id=round_id,
                open_gap_count=open_gap_count,
                planned_gap_count=planned_gap_count,
                candidate_count=candidate_count,
                accepted_candidate_count=accepted_candidate_count,
                new_paper_count=new_paper_count,
                integrated_unit_count=integrated_unit_count,
                paper_count_before=paper_count_before,
                paper_count_after=paper_count_after,
                group_count_before=group_count_before,
                group_count_after=group_count_after,
                round_outcome=round_outcome,
                stop_reason=stop_reason,
            )
        )

    def _append_group_retrieval_attempts(
        self,
        attempts_by_group: dict[str, list[dict]],
        gaps,
        candidates: list[dict],
        accepted: list[dict],
        rejected: list[dict],
        integrations,
        round_id: int,
    ) -> None:
        for gap in gaps:
            gap_candidates = [item for item in candidates if item.get("gap_id") == gap.gap_id]
            gap_accepted = [item for item in accepted if item["gap"].gap_id == gap.gap_id]
            gap_rejected = [item for item in rejected if item.get("gap_id") == gap.gap_id]
            gap_integrations = [item for item in integrations if item.gap_id == gap.gap_id]
            attempts_by_group.setdefault(gap.group_id, []).append(
                {
                    "round_id": round_id,
                    "gap_description": gap.description,
                    "retrieval_query": gap.retrieval_query,
                    "candidate_count": len(gap_candidates),
                    "accepted_count": len(gap_accepted),
                    "integrated_unit_count": len(gap_integrations),
                    "rejection_reason_counts": dict(Counter(item.get("reason", "unknown") for item in gap_rejected)),
                }
            )
