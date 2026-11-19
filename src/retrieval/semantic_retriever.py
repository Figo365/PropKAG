from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.contribution.schemas import Paper
from src.utils.io import read_json


DEFAULT_SEARCH_TYPE = "semantic"


class RetrievalError(RuntimeError):
    """Raised when no trustworthy retrieval source can provide usable papers."""


class SemanticRetriever:
    def __init__(
        self,
        input_path: Path,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        search_type: str | None = None,
        timeout_seconds: int = 30,
    ):
        self.input_path = input_path
        self.api_endpoint = (api_endpoint or os.getenv("LITERATURE_SEARCH_API_ENDPOINT") or "").strip().rstrip("/")
        self.api_key = (api_key or os.getenv("LITERATURE_SEARCH_API_KEY") or "").strip()
        self.search_type = search_type or os.getenv("LITERATURE_SEARCH_TYPE") or DEFAULT_SEARCH_TYPE
        self.timeout_seconds = timeout_seconds
        self.last_response: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_source = "unknown"

    def search(self, query: str, top_k: int = 50) -> list[Paper]:
        self.last_error = None
        if self.api_key or self.api_endpoint:
            if not (self.api_key and self.api_endpoint):
                self.last_source = "remote_api_misconfigured"
                self.last_error = (
                    "Literature search API is incomplete. Set both "
                    "LITERATURE_SEARCH_API_KEY and LITERATURE_SEARCH_API_ENDPOINT."
                )
                raise RetrievalError(self.last_error)
            papers = self._search_remote(query, top_k)
            if papers:
                self.last_source = "remote_api"
                return papers
            self.last_source = "remote_api_failed"
            raise RetrievalError(self.last_error or "Remote semantic retrieval returned no usable papers.")

        try:
            raw = read_json(self.input_path, default=None)
        except (OSError, json.JSONDecodeError) as exc:
            self.last_source = "local_input_failed"
            self.last_error = f"Local retrieval input could not be read: {exc}"
            raise RetrievalError(self.last_error) from exc
        if raw is None:
            self.last_source = "unavailable"
            self.last_error = (
                "No literature source is configured. Set LITERATURE_SEARCH_API_KEY and "
                "LITERATURE_SEARCH_API_ENDPOINT "
                f"or provide a local paper list at {self.input_path}."
            )
            raise RetrievalError(self.last_error)
        if not isinstance(raw, list) or not raw:
            self.last_source = "local_input_failed"
            self.last_error = f"Local retrieval input must be a non-empty JSON array: {self.input_path}."
            raise RetrievalError(self.last_error)

        self.last_source = "local_input"
        fields = Paper.__dataclass_fields__
        try:
            papers = [Paper(**{key: value for key, value in item.items() if key in fields}) for item in raw]
        except (AttributeError, TypeError, ValueError) as exc:
            self.last_source = "local_input_failed"
            self.last_error = f"Local retrieval input contains an invalid paper record: {exc}"
            raise RetrievalError(self.last_error) from exc
        query_terms = self._query_terms(query)
        if not query_terms:
            self.last_error = "The retrieval query contains no usable terms."
            raise RetrievalError(self.last_error)
        matched: list[Paper] = []
        for paper in papers:
            haystack = " ".join([paper.title, paper.abstract, " ".join(paper.keywords)]).lower()
            overlap = sum(1 for term in query_terms if term in haystack)
            if overlap == 0:
                continue
            paper.retrieval_score = round(overlap / len(query_terms), 4)
            matched.append(paper)
        if not matched:
            self.last_error = "The local paper list contains no papers matching the retrieval query."
            raise RetrievalError(self.last_error)
        return sorted(matched, key=lambda paper: paper.retrieval_score, reverse=True)[:top_k]

    def _search_remote(self, query: str, limit: int) -> list[Paper]:
        from urllib.parse import urlencode
        query_params = urlencode({
            "api_key": self.api_key,
            "search_type": self.search_type,
            "query": query,
            "limit": limit,
        })
        separator = "&" if "?" in self.api_endpoint else "?"
        url = f"{self.api_endpoint}{separator}{query_params}"
        request = urllib.request.Request(
            url,
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            verify_ssl = os.getenv("LITERATURE_API_VERIFY_SSL", "true").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            request_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
            if not verify_ssl:
                request_kwargs["context"] = ssl._create_unverified_context()
            with urllib.request.urlopen(request, **request_kwargs) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_error = f"Remote semantic retrieval failed: {exc}"
            self.last_response = {"error": self.last_error, "request": {"api_key": "***", "url": url.split("api_key=")[0] + "api_key=***"}}
            return []

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            self.last_error = f"Remote semantic retrieval returned invalid JSON: {exc}"
            self.last_response = {
                "error": self.last_error,
                "body": body[:1000],
                "request": {"url": url.split("api_key=")[0] + "api_key=***"},
            }
            return []

        self.last_response = self._mask_api_key(data, {"url": url.split("api_key=")[0] + "api_key=***"})
        if data.get("code") != 200:
            self.last_error = f"Remote semantic retrieval returned code={data.get('code')}: {data.get('message')}"
            return []

        results = data.get("data", {}).get("results", [])
        if not isinstance(results, list):
            self.last_error = "Remote semantic retrieval response missing data.results list."
            return []

        try:
            papers = [self._result_to_paper(item, idx) for idx, item in enumerate(results, start=1)]
        except (AttributeError, TypeError, ValueError) as exc:
            self.last_error = f"Remote semantic retrieval returned an invalid paper record: {exc}"
            return []
        usable = [paper for paper in papers if paper.title and paper.abstract]
        if not usable:
            self.last_error = "Remote semantic retrieval returned no usable papers with both title and abstract."
        return usable

    def _query_terms(self, query: str) -> set[str]:
        english = {term for term in re.findall(r"[a-zA-Z0-9]+", query.casefold()) if len(term) > 1}
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", query)
        chinese = {
            run[index : index + 2]
            for run in chinese_runs
            for index in range(max(len(run) - 1, 0))
        }
        return english | chinese

    def _result_to_paper(self, item: dict[str, Any], idx: int) -> Paper:
        publication_date = str(item.get("publication_date") or "")
        return Paper(
            paper_id=f"P{idx:03d}",
            title=str(item.get("title") or "").strip(),
            abstract=self._clean_abstract(str(item.get("abstract") or "")),
            keywords=self._normalize_keywords(item.get("keywords")),
            authors=self._normalize_authors(item.get("authors")),
            year=self._parse_year(publication_date),
            journal=str(item.get("source") or ""),
            doi=item.get("doi"),
            url=item.get("url"),
            publication_date=publication_date or None,
            retrieval_score=float(item.get("relevance_score") or 0.0),
            source="semantic_api",
        )

    def _mask_api_key(self, data: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        masked = dict(data)
        masked["request"] = {**payload, "api_key": "***"}
        return masked

    def _normalize_keywords(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in re.split(r"[;；,，]", str(value)) if v.strip()]

    def _normalize_authors(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            normalized: list[str] = []
            for author in value:
                normalized.extend([part.strip() for part in re.split(r"[;；]", str(author)) if part.strip()])
            return normalized
        return [v.strip() for v in re.split(r"[;；,，]", str(value)) if v.strip()]

    def _parse_year(self, publication_date: str) -> int:
        match = re.search(r"(19|20)\d{2}", publication_date)
        return int(match.group(0)) if match else 0

    def _clean_abstract(self, abstract: str) -> str:
        return abstract.replace("△ Less", "").strip()
