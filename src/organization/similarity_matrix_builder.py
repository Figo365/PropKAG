from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone

from src.contribution.schemas import KnowledgeUnit, SimilarityMatrix
from src.utils.embedding_client import EmbeddingClient


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def vector_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SimilarityMatrixBuilder:
    fallback_model = "local_token_cosine_v1"

    def __init__(self, embedding_client: EmbeddingClient | None = None):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.last_status: dict = {
            "embedding_enabled": self.embedding_client.enabled,
            "mode": "not_started",
            "model": None,
            "dimensions": None,
            "error": None,
        }

    def build(self, unit_type: str, units: list[KnowledgeUnit]) -> SimilarityMatrix:
        if self.embedding_client.enabled and units:
            try:
                return self._build_with_embeddings(unit_type, units)
            except Exception as exc:
                self.last_status = {
                    "embedding_enabled": True,
                    "mode": "fallback_token_cosine",
                    "model": self.embedding_client.model,
                    "dimensions": self.embedding_client.dimensions,
                    "error": str(exc),
                }
        return self._build_with_token_cosine(unit_type, units)

    def _build_with_embeddings(self, unit_type: str, units: list[KnowledgeUnit]) -> SimilarityMatrix:
        result = self.embedding_client.embed_texts([unit.text for unit in units])
        for unit, vector in zip(units, result.vectors):
            unit.embedding = vector
        matrix: list[list[float]] = []
        for i, vec_i in enumerate(result.vectors):
            row = []
            for j, vec_j in enumerate(result.vectors):
                row.append(1.0 if i == j else round(vector_cosine(vec_i, vec_j), 4))
            matrix.append(row)
        self.last_status = {
            "embedding_enabled": True,
            "mode": "embedding_cosine",
            "model": result.model,
            "dimensions": result.dimensions,
            "error": None,
            "usage": result.usage,
        }
        return SimilarityMatrix(
            unit_type=unit_type,
            unit_ids=[unit.unit_id for unit in units],
            matrix=matrix,
            embedding_model=f"{result.model}:{result.dimensions}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _build_with_token_cosine(self, unit_type: str, units: list[KnowledgeUnit]) -> SimilarityMatrix:
        vectors = [Counter(tokenize(unit.text)) for unit in units]
        matrix: list[list[float]] = []
        for i, vec_i in enumerate(vectors):
            row = []
            for j, vec_j in enumerate(vectors):
                row.append(1.0 if i == j else round(cosine(vec_i, vec_j), 4))
            matrix.append(row)
        if self.last_status.get("mode") == "not_started":
            self.last_status = {
                "embedding_enabled": self.embedding_client.enabled,
                "mode": "token_cosine",
                "model": self.fallback_model,
                "dimensions": None,
                "error": None,
            }
        return SimilarityMatrix(
            unit_type=unit_type,
            unit_ids=[unit.unit_id for unit in units],
            matrix=matrix,
            embedding_model=self.fallback_model,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
