from __future__ import annotations

import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from src.contribution.schemas import Paper


def paper_identity_key(paper: Paper) -> str:
    """Return the primary identity key: canonical URL, then DOI, then title."""
    keys = paper_identity_keys(paper)
    return keys[0] if keys else ""


def paper_identity_keys(paper: Paper) -> tuple[str, ...]:
    """Return one URL-first identity key, falling back only when URL is absent."""
    canonical_url = _canonical_url(paper.url)
    if canonical_url:
        return (f"url:{canonical_url}",)
    normalized_doi = _normalize_doi(paper.doi)
    if normalized_doi:
        return (f"doi:{normalized_doi}",)
    normalized_title = _normalize(paper.title)
    if normalized_title:
        return (f"title:{normalized_title}",)
    return ()


def _canonical_url(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    host = (parsed.hostname or "").casefold()
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/{2,}", "/", unquote(parsed.path or "/")).rstrip("/") or "/"
    if host in {"doi.org", "dx.doi.org"}:
        doi = _normalize_doi(path.lstrip("/"))
        return f"https://doi.org/{doi}" if doi else ""
    if host == "arxiv.org":
        match = re.match(r"/(?:abs|pdf)/([^/]+?)(?:\.pdf)?$", path, flags=re.IGNORECASE)
        if match:
            path = f"/abs/{match.group(1)}"
    query_items = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_parameter(key)
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit(("https", host, path, query, ""))


def _is_tracking_parameter(key: str) -> bool:
    normalized = key.casefold()
    return normalized.startswith("utm_") or normalized in {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "source",
    }


def _normalize_doi(value: str | None) -> str:
    raw = str(value or "").strip().casefold()
    raw = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw)
    raw = re.sub(r"^doi:\s*", "", raw)
    return raw.strip().rstrip("/")


def _normalize(value: str) -> str:
    return re.sub(r"[^\w]+", "", str(value).casefold())
