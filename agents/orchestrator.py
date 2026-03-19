"""agents/orchestrator.py

Coordinates the full query pipeline end-to-end.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import structlog

from agents.vector_agent import VectorAgent
from agents.sql_agent import SQLAgent
from agents.graph_agent import GraphAgent
from agents.contextual_summariser import ContextualSummariser, ScoredSummary
from agents.reasoning_agent import ReasoningAgent, ReasoningResult
from agents.critic_agent import CriticAgent, CriticResult
from cache.redis_cache import RedisCache
from store.chroma_store import SearchResult

log = structlog.get_logger()


@dataclass
class QueryResponse:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    flagged: bool
    unsupported_claims: List[str]
    graph_entities: List[str]
    tool_calls: List[Dict[str, Any]]
    latency: Dict[str, int]
    from_cache: bool = False


class Orchestrator:
    """Central coordinator for the multi-agent RAG pipeline."""

    def __init__(
        self,
        vector_agent: VectorAgent,
        sql_agent: SQLAgent,
        graph_agent: GraphAgent,
        summariser: ContextualSummariser,
        reasoning_agent: ReasoningAgent,
        critic_agent: CriticAgent,
        cache: RedisCache,
    ) -> None:
        self.vector = vector_agent
        self.sql = sql_agent
        self.graph = graph_agent
        self.summariser = summariser
        self.reasoner = reasoning_agent
        self.critic = critic_agent
        self.cache = cache

    async def query(self, query: str) -> QueryResponse:
        pipeline_start = time.perf_counter()
        latency: Dict[str, int] = {}

        cached = await self.cache.get(query)
        if cached:
            log.info("cache_hit", query=query[:80])
            cached["from_cache"] = True
            return QueryResponse(**cached)

        # 2. Parallel retrieval
        t = time.perf_counter()
        vector_result, sql_result, graph_result = await asyncio.gather(
            self.vector.search(query),
            self.sql.search(query),
            self.graph.search(query),
        )
        latency["retrieval_ms"] = round((time.perf_counter() - t) * 1000.0)

        # Currently we only use vector_result.results as chunks;
        # SQL data lands in Chroma via CSV ingestion.
        all_chunks: List[SearchResult] = vector_result.results

        # 4. Contextual summarisation
        t = time.perf_counter()
        summaries: List[ScoredSummary] = await self.summariser.summarise(
            query, all_chunks
        )
        latency["summarisation_ms"] = round((time.perf_counter() - t) * 1000.0)

        # 5. Reasoning
        t = time.perf_counter()
        reasoning: ReasoningResult = await self.reasoner.reason(query, summaries)
        latency["reasoning_ms"] = round(reasoning.latency_ms)

        # 6. Critic
        t = time.perf_counter()
        critic: CriticResult = await self.critic.critique(
            query, summaries, reasoning.answer
        )
        latency["critic_ms"] = round(critic.latency_ms)
        latency["total_ms"] = round((time.perf_counter() - pipeline_start) * 1000.0)

        sources = [
            {
                "source": s.source,
                "page": s.page,
                "relevance_score": s.relevance_score,
            }
            for s in summaries
        ]

        response = QueryResponse(
            query=query,
            answer=reasoning.answer,
            sources=sources,
            confidence=critic.confidence,
            flagged=critic.flagged,
            unsupported_claims=critic.unsupported_claims,
            graph_entities=graph_result.related_entities[:10],
            tool_calls=reasoning.tool_calls,
            latency=latency,
        )

        await self.cache.set(query, response.__dict__)
        log.info("query_complete", latency=latency, confidence=critic.confidence)
        return response

