from __future__ import annotations

import os

from src.agents.base import BaseAgent
from src.contribution.schemas import ContributionRecord, KnowledgeUnit
from src.organization.community_detector import CommunityDetector
from src.organization.group_synthesis_input_builder import GroupContextBuilder
from src.organization.group_table_builder import GroupContributionTableBuilder
from src.organization.similarity_matrix_builder import SimilarityMatrixBuilder
from src.organization.unit_group_builder import UnitGroupBuilder
from src.organization.unit_group_namer import UnitGroupNamer
from src.organization.unit_relation_builder import UnitRelationBuilder
from src.utils.io import write_json


UNIT_TYPES = ["issue", "problem", "method", "finding"]


class KnowledgeOrganizationAgent(BaseAgent):
    name = "KnowledgeOrganizationAgent"

    def run(
        self,
        units: list[KnowledgeUnit],
        records: list[ContributionRecord],
        graph_threshold: float,
        review_topic: str = "",
        prompts: dict[str, str] | None = None,
    ):
        task = self.state.start_task(
            self.name,
            ["knowledge_units.json", "contribution_records.json", "papers.json", "prompts_used.json"],
        )
        records_by_id = {r.paper_id: r for r in records}
        units_by_id = {u.unit_id: u for u in units}
        matrix_builder = SimilarityMatrixBuilder()
        leiden_resolution = max(0.01, float(os.getenv("GROUP_LEIDEN_RESOLUTION", "1.0")))
        leiden_seed = int(os.getenv("GROUP_LEIDEN_SEED", "42"))
        community_backend = "leiden"
        relation_builder = UnitRelationBuilder(graph_threshold)
        graph_config = {
            "edge_rule": "threshold",
            "similarity_threshold": graph_threshold,
            "resolution": leiden_resolution,
            "random_seed": leiden_seed,
            "backend": community_backend,
        }
        detector = CommunityDetector(
            resolution=leiden_resolution,
            random_seed=leiden_seed,
            backend=community_backend,
        )
        group_builder = UnitGroupBuilder(graph_threshold)
        table_builder = GroupContributionTableBuilder()
        group_context_builder = GroupContextBuilder()
        all_relations = []
        all_accepted_relations = []
        detected_communities = {}
        all_groups = []
        diagnostics = []
        embedding_status = {}

        outputs = []
        for unit_type in UNIT_TYPES:
            same_type = [u for u in units if u.unit_type == unit_type]
            matrix = matrix_builder.build(unit_type, same_type)
            embedding_status[unit_type] = matrix_builder.last_status
            outputs.append(write_json(self.output_dir / f"similarity_matrix_{unit_type}.json", matrix))
            candidate_relations = relation_builder.build_candidates(matrix)
            relations = relation_builder.accept(candidate_relations)
            accepted_pairs = {
                (relation.source_unit, relation.target_unit): relation
                for relation in relations
            }
            for relation in candidate_relations:
                if (relation.source_unit, relation.target_unit) in accepted_pairs:
                    relation.status = "accepted"
                    relation.reason = accepted_pairs[(relation.source_unit, relation.target_unit)].reason
                else:
                    relation.status = "rejected"
                    relation.reason = f"below_similarity_threshold:{graph_threshold:.4f}"
            all_relations.extend(candidate_relations)
            all_accepted_relations.extend(relations)
            unit_ids = [u.unit_id for u in same_type]
            communities, modularity = detector.detect(
                unit_ids,
                relations,
            )
            semantic_communities = [community for community in communities if len(community) >= 2]
            singleton_units = {
                unit_id
                for community in communities
                if len(community) == 1
                for unit_id in community
            }
            grouped_communities = [*semantic_communities]
            if singleton_units:
                grouped_communities.append(singleton_units)
            groups = group_builder.build(
                unit_type,
                grouped_communities,
                modularity,
                units_by_id,
                records_by_id,
                relations,
                detector.algorithm,
                other_unit_ids=singleton_units,
            )
            all_groups.extend(groups)
            detected_communities[unit_type] = {
                "algorithm": detector.algorithm,
                "modularity": modularity,
                "graph_config": graph_config,
                "raw_communities": [sorted(community) for community in communities],
                "semantic_communities": [sorted(community) for community in semantic_communities],
                "singleton_units": sorted(singleton_units),
                "other_group_id": f"{unit_type.upper()}G999" if singleton_units else None,
            }
            diagnostics.extend(
                {
                    "group_id": g.group_id,
                    "group_type": g.group_type,
                    "size": g.size,
                    "density": g.density,
                    "mean_similarity": g.mean_similarity,
                    "min_similarity": g.min_similarity,
                    "quality_flag": g.quality_flag,
                    "group_role": g.group_role,
                }
                for g in groups
            )

        group_namer = UnitGroupNamer(prompts)
        all_groups, group_naming_status = group_namer.name(all_groups, units_by_id, review_topic)
        tables = [table_builder.build(group, records_by_id, units_by_id) for group in all_groups]
        group_contexts = [group_context_builder.build(review_topic, group, records_by_id, units_by_id) for group in all_groups]
        outputs.extend(
            [
                write_json(self.output_dir / "unit_relations.json", all_relations),
                write_json(self.output_dir / "detected_communities.json", detected_communities),
                write_json(self.output_dir / "unit_groups.json", all_groups),
                write_json(self.output_dir / "unit_group_diagnostics.json", diagnostics),
                write_json(self.output_dir / "group_naming_llm_status.json", group_naming_status),
                write_json(self.output_dir / "group_contribution_tables.json", tables),
                write_json(self.output_dir / "group_contexts.json", group_contexts),
                write_json(self.output_dir / "embedding_status.json", embedding_status),
            ]
        )
        self.state.finish_task(task, outputs)
        self.state.add_message(
            self.name,
            "SynthesisAgent",
            "organization_completed",
            {
                "group_count": len(all_groups),
                "relation_count": len(all_relations),
                "accepted_relation_count": len(all_accepted_relations),
                "embedding_status": embedding_status,
                "grouping_config": graph_config,
            },
            task.task_id,
        )
        return all_groups, tables, group_contexts
