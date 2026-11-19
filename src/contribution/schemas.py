from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    keywords: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    year: int = 0
    journal: str = ""
    doi: Optional[str] = None
    url: Optional[str] = None
    publication_date: Optional[str] = None
    retrieval_score: float = 0.0
    source: str = "local"
    retrieval_round: int = 0
    gap_id: Optional[str] = None
    query_id: Optional[str] = None
    admission_reason: Optional[str] = None


@dataclass
class GroupGap:
    gap_id: str
    round_id: int
    group_id: str
    group_type: str
    need_retrieval: bool
    description: str
    retrieval_query: str = ""
    decision: str = "retrieval_required"


@dataclass
class RetrievalPlan:
    plan_id: str
    round_id: int
    gap_id: str
    group_id: str
    query: str


@dataclass
class RoundMetrics:
    round_id: int
    open_gap_count: int
    planned_gap_count: int
    candidate_count: int
    accepted_candidate_count: int
    new_paper_count: int
    integrated_unit_count: int
    paper_count_before: int
    paper_count_after: int
    group_count_before: int
    group_count_after: int
    round_outcome: str = ""
    stop_reason: str = ""


@dataclass
class GroupIntegration:
    integration_id: str
    round_id: int
    gap_id: str
    target_group_id: str
    paper_id: str
    query_id: str
    reused_paper: bool
    unit_type: str
    unit_text: str
    source_evidence: str
    integration_role: str
    decision_reason: str


@dataclass
class ContributionRecord:
    paper_id: str
    issues: list[str]
    research_problems: list[str]
    research_methods: list[str]
    research_findings: list[str]
    evidence_text: dict[str, str]
    paper_summary: str = ""


@dataclass
class KnowledgeUnit:
    unit_id: str
    paper_id: str
    unit_type: str
    text: str
    evidence_text: Optional[str] = None
    paper_summary: str = ""
    embedding: Optional[list[float]] = None


@dataclass
class SimilarityMatrix:
    unit_type: str
    unit_ids: list[str]
    matrix: list[list[float]]
    embedding_model: str
    created_at: str


@dataclass
class UnitRelation:
    source_unit: str
    target_unit: str
    unit_type: str
    relation_type: str
    similarity: float
    confidence: float
    status: str = "accepted"
    verification_method: str = "similarity_only"
    reason: Optional[str] = None


@dataclass
class UnitGroup:
    group_id: str
    group_type: str
    label: str
    units: list[str]
    representative_papers: list[str]
    summary: str
    size: int
    edge_count: int
    density: float
    mean_similarity: float
    min_similarity: float
    modularity: Optional[float]
    community_algorithm: str
    quality_flag: str
    structure_scope: str = "initial_grouping"
    initial_size: Optional[int] = None
    incremental_unit_count: int = 0
    group_role: str = "semantic"


@dataclass
class GroupUnitRef:
    unit_id: str
    unit_type: str
    text: str


@dataclass
class GroupContribution:
    paper_id: str
    paper_summary: str
    units: list[GroupUnitRef] = field(default_factory=list)


@dataclass
class GroupContributionTable:
    group_id: str
    group_type: str
    group_label: str
    group_summary: str
    group_quality_flag: str
    contributions: list[GroupContribution]
    group_role: str = "semantic"


@dataclass
class GroupContextUnit:
    """One group member shared by group synthesis and gap diagnosis."""

    unit_id: str
    paper_id: str
    text: str
    background: str


@dataclass
class GroupContext:
    """Single group-level context shared by gap retrieval and group synthesis."""

    group_id: str
    group_type: str
    group_label: str
    group_summary: str
    group_quality_flag: str
    size: int
    review_topic: str
    units: list[GroupContextUnit] = field(default_factory=list)
    structure_scope: str = "initial_grouping"
    initial_size: Optional[int] = None
    incremental_unit_count: int = 0
    group_role: str = "semantic"


@dataclass
class ReviewFragment:
    fragment_id: str
    group_id: str
    group_type: str
    title: str
    text: str
    cited_papers: list[str]
    group_role: str = "semantic"


@dataclass
class TypeSynthesisSentence:
    """One cited statement inside a type-level review paragraph."""

    text: str
    source_fragment_ids: list[str]
    cited_papers: list[str]


@dataclass
class TypeSynthesisParagraph:
    relation_focus: str
    text: str
    source_fragment_ids: list[str]
    cited_papers: list[str]
    role: str = "exposition"
    sentences: list[TypeSynthesisSentence] = field(default_factory=list)


@dataclass
class TypeSynthesis:
    type_synthesis_id: str
    group_type: str
    title: str
    text: str
    cited_papers: list[str]
    covered_fragment_ids: list[str]
    uncovered_fragment_ids: list[str] = field(default_factory=list)
    paragraphs: list[TypeSynthesisParagraph] = field(default_factory=list)
    quality_issues: list[dict] = field(default_factory=list)
