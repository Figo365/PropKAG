from __future__ import annotations

from src.contribution.schemas import SimilarityMatrix, UnitRelation


class UnitRelationBuilder:
    def __init__(self, graph_threshold: float = 0.60):
        self.graph_threshold = graph_threshold

    def build_candidates(self, similarity: SimilarityMatrix) -> list[UnitRelation]:
        relations: list[UnitRelation] = []
        ids = similarity.unit_ids
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = similarity.matrix[i][j]
                relations.append(
                    UnitRelation(
                        source_unit=ids[i],
                        target_unit=ids[j],
                        unit_type=similarity.unit_type,
                        relation_type=f"{similarity.unit_type}_related",
                        similarity=sim,
                        confidence=sim,
                        status="candidate",
                        reason="similarity_evaluated",
                    )
                )
        return relations

    def accept(self, candidates: list[UnitRelation]) -> list[UnitRelation]:
        accepted: list[UnitRelation] = []
        for relation in candidates:
            if relation.similarity < self.graph_threshold:
                continue
            accepted.append(
                UnitRelation(
                    source_unit=relation.source_unit,
                    target_unit=relation.target_unit,
                    unit_type=relation.unit_type,
                    relation_type=relation.relation_type,
                    similarity=relation.similarity,
                    confidence=relation.confidence,
                    status="accepted",
                    verification_method=relation.verification_method,
                    reason=f"similarity >= {self.graph_threshold:.4f}",
                )
            )
        return accepted

    def build(self, similarity: SimilarityMatrix) -> list[UnitRelation]:
        """Compatibility entry point returning graph-ready accepted edges."""
        return self.accept(self.build_candidates(similarity))
