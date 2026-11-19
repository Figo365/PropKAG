from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_MAX_CHARS = 2000


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dimensions: int
    usage: dict[str, Any] | None = None


class EmbeddingClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout_seconds: int = 60,
        max_retries: int = 1,
        batch_size: int | None = None,
        max_chars: int | None = None,
    ):
        self.api_key = (api_key or os.getenv("EMBEDDING_API_KEY") or "").strip()
        self.base_url = (base_url or os.getenv("EMBEDDING_API_BASE_URL") or "").strip().rstrip("/")
        self.model = (model or os.getenv("EMBEDDING_MODEL") or "").strip()
        self.dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS") or DEFAULT_EMBEDDING_DIMENSIONS)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.batch_size = max(1, batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE") or DEFAULT_EMBEDDING_BATCH_SIZE))
        self.max_chars = max(100, max_chars or int(os.getenv("EMBEDDING_MAX_CHARS") or DEFAULT_EMBEDDING_MAX_CHARS))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not self.enabled:
            raise RuntimeError(
                "EmbeddingClient is not configured. Set EMBEDDING_API_KEY, "
                "EMBEDDING_API_BASE_URL, and EMBEDDING_MODEL."
            )
        if not texts:
            return EmbeddingResult([], self.model, self.dimensions, usage=None)

        normalized_texts = [self._normalize_text(text) for text in texts]
        vectors: list[list[float]] = []
        total_usage: dict[str, Any] = {}
        model = self.model
        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]
            result = self._embed_batch_resilient(batch)
            vectors.extend(result.vectors)
            model = result.model
            if result.usage:
                for key, value in result.usage.items():
                    if isinstance(value, (int, float)):
                        total_usage[key] = total_usage.get(key, 0) + value
        return EmbeddingResult(
            vectors=vectors,
            model=model,
            dimensions=len(vectors[0]) if vectors else self.dimensions,
            usage=total_usage or None,
        )

    def _embed_batch_resilient(self, texts: list[str]) -> EmbeddingResult:
        try:
            return self._embed_batch(texts)
        except RuntimeError:
            if len(texts) <= 1:
                raise
            midpoint = len(texts) // 2
            left = self._embed_batch_resilient(texts[:midpoint])
            right = self._embed_batch_resilient(texts[midpoint:])
            usage = self._merge_usage(left.usage, right.usage)
            return EmbeddingResult(
                vectors=left.vectors + right.vectors,
                model=right.model or left.model,
                dimensions=left.dimensions or right.dimensions,
                usage=usage,
            )

    def _embed_batch(self, texts: list[str]) -> EmbeddingResult:
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        body = self._send_with_retry(request)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Embedding API returned invalid JSON: {body[:1000]}") from exc

        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError(f"Embedding API response missing data list: {data}")

        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        vectors = [self._validate_vector(item.get("embedding")) for item in ordered]
        return EmbeddingResult(
            vectors=vectors,
            model=str(data.get("model") or self.model),
            dimensions=len(vectors[0]) if vectors else self.dimensions,
            usage=data.get("usage"),
        )

    def _validate_vector(self, value: Any) -> list[float]:
        if not isinstance(value, list) or not value:
            raise RuntimeError("Embedding item missing non-empty embedding vector.")
        vector = [float(x) for x in value]
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            raise RuntimeError("Embedding API returned zero vector.")
        return vector

    def _normalize_text(self, text: str) -> str:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            normalized = "[empty]"
        return normalized[: self.max_chars]

    def _merge_usage(self, left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        for usage in [left, right]:
            if not usage:
                continue
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    merged[key] = merged.get(key, 0) + value
        return merged or None

    def _send_with_retry(self, request: urllib.request.Request) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Embedding request failed after {self.max_retries + 1} attempt(s): {last_exc}")
