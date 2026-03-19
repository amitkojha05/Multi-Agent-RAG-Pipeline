"""api/models.py — Pydantic schemas for all API endpoints."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


class SourceCitation(BaseModel):
    source: str
    page: Optional[int] = None
    relevance_score: int


class LatencyBreakdown(BaseModel):
    retrieval_ms: int
    summarisation_ms: int
    reasoning_ms: int
    critic_ms: int
    total_ms: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    confidence: float
    flagged: bool
    unsupported_claims: List[str]
    graph_entities: List[str]
    tool_calls: List[Dict[str, Any]]
    latency: LatencyBreakdown
    from_cache: bool


class IngestRequest(BaseModel):
    source_type: str = Field(..., pattern="^(pdf|csv)$")
    file_path: str


class IngestResponse(BaseModel):
    status: str
    source: str
    details: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    chroma_docs: int
    pg_tables: List[str]
    graph_nodes: int
    graph_edges: int
    cache_connected: bool

