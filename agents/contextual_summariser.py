"""agents/contextual_summariser.py

ContextualSummariser evaluates each retrieved chunk for relevance
to the query and produces a scored summary.

Only chunks with score >= SUMMARISER_SCORE_THRESHOLD pass through.
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import anthropic
import structlog

from store.chroma_store import SearchResult

log = structlog.get_logger()


@dataclass
class ScoredSummary:
    chunk_id: str
    source: str
    page: Optional[int]
    relevance_score: int
    summary: str
    original_similarity: float


SUMMARISE_PROMPT_TEMPLATE = """You are a scientific research assistant.
You are given a user query and a text chunk retrieved via semantic search.

Your job:
1. Decide how relevant this chunk is to answering the query, on a scale of 1-5:
   1 = completely irrelevant
   2 = mostly irrelevant / tangential
   3 = somewhat relevant
   4 = clearly relevant
   5 = highly relevant / directly answers the query
2. If relevance_score >= 3, write a short summary (2-4 sentences)
   of the information in this chunk that is useful for answering the query.

Return ONLY a JSON object with keys:
  "relevance_score": <integer 1-5>
  "summary": "<short summary, or empty string if score < 3>"

Query:
{query}

Source: {source}, page {page}

Chunk:
\"\"\"{chunk_text}\"\"\""""


class ContextualSummariser:
    """Scores and summarises retrieved chunks in context of the query."""

    def __init__(
        self,
        score_threshold: int = 3,
        max_concurrent: int = 4,
        model: Optional[str] = None,
    ) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.score_threshold = int(
            os.getenv("SUMMARISER_SCORE_THRESHOLD", str(score_threshold))
        )
        self.max_concurrent = max_concurrent

    async def summarise(
        self,
        query: str,
        chunks: List[SearchResult],
    ) -> List[ScoredSummary]:
        t0 = time.perf_counter()
        if not chunks:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def score_one(result: SearchResult) -> Optional[ScoredSummary]:
            async with semaphore:
                return await self._score_chunk(query, result)

        scored = await asyncio.gather(*[score_one(r) for r in chunks])

        passing = [
            s for s in scored if s is not None and s.relevance_score >= self.score_threshold
        ]
        passing.sort(key=lambda s: s.relevance_score, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000.0
        log.info(
            "summarisation_complete",
            total=len(chunks),
            passing=len(passing),
            latency_ms=round(elapsed),
        )
        return passing

    async def _score_chunk(
        self, query: str, result: SearchResult
    ) -> Optional[ScoredSummary]:
        chunk = result.chunk
        prompt = SUMMARISE_PROMPT_TEMPLATE.format(
            query=query,
            source=chunk.source,
            page=chunk.page or "N/A",
            chunk_text=chunk.text[:1500],
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
            return ScoredSummary(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page=chunk.page,
                relevance_score=int(data["relevance_score"]),
                summary=data.get("summary", ""),
                original_similarity=result.score,
            )
        except Exception as e:  # pragma: no cover - logged and skipped
            log.warning(
                "summarise_chunk_failed",
                error=str(e),
                chunk_id=getattr(chunk, "chunk_id", "?"),
            )
            return None

