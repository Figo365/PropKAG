from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path

from src.contribution.schemas import UnitRelation


def build_adjacency(unit_ids: list[str], relations: list[UnitRelation]) -> dict[str, dict[str, float]]:
    graph: dict[str, dict[str, float]] = {unit_id: {} for unit_id in unit_ids}
    for relation in relations:
        if relation.source_unit not in graph or relation.target_unit not in graph:
            continue
        graph[relation.source_unit][relation.target_unit] = relation.similarity
        graph[relation.target_unit][relation.source_unit] = relation.similarity
    return graph


def weighted_modularity(communities: list[set[str]], graph: dict[str, dict[str, float]]) -> float:
    total_weight = sum(sum(neighbours.values()) for neighbours in graph.values()) / 2
    if total_weight == 0:
        return 0.0
    degree = {node: sum(neighbours.values()) for node, neighbours in graph.items()}
    score = 0.0
    for community in communities:
        for left in community:
            for right in community:
                edge_weight = graph.get(left, {}).get(right, 0.0)
                score += edge_weight - degree[left] * degree[right] / (2 * total_weight)
    return round(score / (2 * total_weight), 6)


class CommunityDetector:
    """Detect one-level semantic communities from accepted weighted edges.

    Formal runs use graspologic's Leiden implementation. A deterministic
    weighted-modularity implementation is retained only for explicit legacy
    and unit-test use; the default never falls back to it.
    """

    def __init__(self, resolution: float = 1.0, random_seed: int = 42, backend: str = "leiden"):
        if backend not in {"auto", "leiden", "modularity"}:
            raise ValueError("backend must be auto, leiden, or modularity")
        self.resolution = max(0.01, float(resolution))
        self.random_seed = int(random_seed)
        self.backend = backend
        self.last_algorithm = "not_run"

    @property
    def algorithm(self) -> str:
        return self.last_algorithm

    def detect(
        self,
        unit_ids: list[str],
        relations: list[UnitRelation],
        **_legacy_options,
    ) -> tuple[list[set[str]], float]:
        ordered_ids = list(dict.fromkeys(unit_ids))
        if not ordered_ids:
            self.last_algorithm = "weighted_modularity_v1" if self.backend == "modularity" else "leiden_v1"
            return [], 0.0

        accepted = [relation for relation in relations if relation.status == "accepted"]
        if self.backend in {"auto", "leiden"}:
            try:
                communities = self._detect_leiden(ordered_ids, accepted)
                self.last_algorithm = "leiden_v1"
                self._validate_partition(ordered_ids, communities)
                return self._ordered(communities), weighted_modularity(
                    communities, build_adjacency(ordered_ids, accepted)
                )
            except (ImportError, ModuleNotFoundError):
                if self.backend == "leiden":
                    raise RuntimeError(
                        "Leiden grouping requires graspologic. Install the project environment first."
                    )

        communities = self._detect_modularity(ordered_ids, accepted)
        self.last_algorithm = "weighted_modularity_v1"
        self._validate_partition(ordered_ids, communities)
        return self._ordered(communities), weighted_modularity(
            communities, build_adjacency(ordered_ids, accepted)
        )

    def _detect_leiden(
        self,
        unit_ids: list[str],
        relations: list[UnitRelation],
    ) -> list[set[str]]:
        runtime_cache = Path(tempfile.gettempdir()) / "alr_knowledge_org_cache"
        for variable, child in {
            "NUMBA_CACHE_DIR": "numba",
            "MPLCONFIGDIR": "matplotlib",
            "XDG_CACHE_HOME": "xdg",
        }.items():
            if variable not in os.environ:
                cache_dir = runtime_cache / child
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ[variable] = str(cache_dir)

        from graspologic.partition import hierarchical_leiden

        if not relations:
            return [{unit_id} for unit_id in unit_ids]
        assignments = hierarchical_leiden(
            [
                (relation.source_unit, relation.target_unit, relation.similarity)
                for relation in relations
            ],
            max_cluster_size=max(2, len(unit_ids) + 1),
            random_seed=self.random_seed,
            resolution=self.resolution,
        )
        minimum_level = min(int(getattr(item, "level")) for item in assignments)
        grouped: dict[str, set[str]] = defaultdict(set)
        for assignment in assignments:
            if int(getattr(assignment, "level")) != minimum_level:
                continue
            grouped[str(getattr(assignment, "cluster"))].add(str(getattr(assignment, "node")))

        assigned = {unit_id for community in grouped.values() for unit_id in community}
        communities = list(grouped.values())
        communities.extend({unit_id} for unit_id in unit_ids if unit_id not in assigned)
        return communities

    def _detect_modularity(
        self,
        unit_ids: list[str],
        relations: list[UnitRelation],
    ) -> list[set[str]]:
        graph = build_adjacency(unit_ids, relations)
        communities = [{unit_id} for unit_id in unit_ids]
        current_score = weighted_modularity(communities, graph)
        while len(communities) > 1:
            best_pair: tuple[int, int] | None = None
            best_score = current_score
            for left_index in range(len(communities)):
                for right_index in range(left_index + 1, len(communities)):
                    if not any(
                        neighbour in communities[right_index]
                        for node in communities[left_index]
                        for neighbour in graph.get(node, {})
                    ):
                        continue
                    merged = [set(community) for community in communities]
                    merged[left_index] |= merged[right_index]
                    del merged[right_index]
                    score = weighted_modularity(merged, graph)
                    if score > best_score + 1e-9:
                        best_score = score
                        best_pair = (left_index, right_index)
            if best_pair is None:
                break
            left_index, right_index = best_pair
            communities[left_index] |= communities[right_index]
            del communities[right_index]
            current_score = best_score
        return communities

    def _validate_partition(self, unit_ids: list[str], communities: list[set[str]]) -> None:
        expected = set(unit_ids)
        flattened = [unit_id for community in communities for unit_id in community]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Community detection assigned a knowledge unit more than once.")
        if set(flattened) != expected:
            missing = sorted(expected - set(flattened))
            extra = sorted(set(flattened) - expected)
            raise ValueError(f"Community partition mismatch; missing={missing}, extra={extra}.")

    def _ordered(self, communities: list[set[str]]) -> list[set[str]]:
        return sorted(communities, key=lambda community: (-len(community), sorted(community)))


def serialize_graph(unit_ids: list[str], relations: list[UnitRelation]) -> dict:
    return {
        "nodes": unit_ids,
        "edges": [
            {
                "source": relation.source_unit,
                "target": relation.target_unit,
                "weight": relation.similarity,
                "unit_type": relation.unit_type,
            }
            for relation in relations
        ],
    }
