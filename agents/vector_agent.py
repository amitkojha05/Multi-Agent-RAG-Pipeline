"""agents/vector_agent.py

VectorAgent performs semantic similarity search over ChromaDB.
Searches both the "papers" collection (PDF chunks) and the
"datasets" collection (CSV row embeddings) concurrently.
"""
import asyncio
import os
import time
from dataclasses import dataclass
from typing import List

from store.chroma_store import ChromaStore, SearchResult


@dataclass
class VectorResult:
    results: List[SearchResult]
    latency_ms: float
    collections_searched: List[str]


class VectorAgent:
    """Semantic search specialist over ChromaDB collections."""

    def __init__(
        self,
        store: ChromaStore,
        top_k: int | None = None,
        paper_collection: str | None = None,
        dataset_collection: str | None = None,
    ) -> None:
        self.store = store
        self.top_k = top_k or int(os.getenv("CHROMA_TOP_K", "8"))
        self.paper_collection = paper_collection or os.getenv(
            "CHROMA_COLLECTION_PAPERS", "papers"
        )
        self.dataset_collection = dataset_collection or os.getenv(
            "CHROMA_COLLECTION_DATASETS", "datasets"
        )

    async def search(self, query: str) -> VectorResult:
        t0 = time.perf_counter()

        paper_count, dataset_count = await asyncio.gather(
            self.store.count(self.paper_collection),
            self.store.count(self.dataset_collection),
        )

        tasks: List[asyncio.Future] = []
        collections: List[str] = []
        if paper_count > 0:
            tasks.append(
                asyncio.create_task(
                    self.store.search(self.paper_collection, query, self.top_k)
                )
            )
            collections.append(self.paper_collection)
        if dataset_count > 0:
            tasks.append(
                asyncio.create_task(
                    self.store.search(self.dataset_collection, query, self.top_k)
                )
            )
            collections.append(self.dataset_collection)

        if not tasks:
            return VectorResult(results=[], latency_ms=0.0, collections_searched=[])

        all_results = await asyncio.gather(*tasks)

        merged: List[SearchResult] = []
        for result_list in all_results:
            merged.extend(result_list)
        merged.sort(key=lambda r: r.score, reverse=True)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return VectorResult(
            results=merged[: self.top_k],
            latency_ms=latency_ms,
            collections_searched=collections,
        )

