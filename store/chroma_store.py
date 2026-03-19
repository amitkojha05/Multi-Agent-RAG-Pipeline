"""store/chroma_store.py

ChromaStore wraps two ChromaDB collections: papers (PDF chunks)
and datasets (CSV row embeddings). Provides upsert, similarity search,
and delete. Embedding is done here, not in the ingestor.
"""
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict

import chromadb
from chromadb import Collection
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    """One retrievable unit stored in ChromaDB."""
    chunk_id: str          # SHA256(source + page + index)
    text: str
    source: str            # filename or dataset name
    page: Optional[int]    # PDF page number (None for CSV rows)
    chunk_index: int       # 0-based position within source
    extra_meta: Dict


@dataclass
class SearchResult:
    chunk: Chunk
    distance: float        # L2 distance (lower = more similar)
    score: float           # 1 - normalised distance (higher = more similar)


class ChromaStore:
    """
    Manages ChromaDB collections for semantic retrieval.

    Usage:
        store = ChromaStore(persist_dir="./chroma_db")
        await store.upsert_chunks("papers", chunks)
        results = await store.search("papers", "enzyme kinetics", top_k=8)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedder = SentenceTransformer(embedding_model)
        self._batch_size = batch_size
        self._collections: Dict[str, Collection] = {}

    def _get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "l2"},
            )
        return self._collections[name]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed in batches to avoid OOM on large ingestions."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._embedder.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    async def upsert_chunks(self, collection_name: str, chunks: List[Chunk]) -> int:
        """Embed and upsert chunks. Returns count upserted."""

        def _sync_upsert() -> int:
            col = self._get_collection(collection_name)
            texts = [c.text for c in chunks]
            embeddings = self._embed(texts)
            col.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source": c.source,
                        "page": c.page if c.page is not None else -1,
                        "chunk_index": c.chunk_index,
                        **c.extra_meta,
                    }
                    for c in chunks
                ],
            )
            return len(chunks)

        return await asyncio.to_thread(_sync_upsert)

    async def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 8,
        where: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Embed query and return top_k most similar chunks."""

        def _sync_search() -> List[SearchResult]:
            col = self._get_collection(collection_name)
            if col.count() == 0:
                return []
            query_embedding = self._embedder.encode(
                [query], normalize_embeddings=True
            ).tolist()
            results = col.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, col.count()),
                where=where,
                include=["documents", "metadatas", "distances", "ids"],
            )
            search_results: List[SearchResult] = []
            for idx, (doc, meta, dist) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                chunk = Chunk(
                    chunk_id=results["ids"][0][idx],
                    text=doc,
                    source=meta.get("source", ""),
                    page=meta.get("page"),
                    chunk_index=meta.get("chunk_index", 0),
                    extra_meta={
                        k: v
                        for k, v in meta.items()
                        if k not in ("source", "page", "chunk_index")
                    },
                )
                # Normalise L2 distance into a similarity-like score.
                score = max(0.0, 1.0 - dist / 2.0)
                search_results.append(
                    SearchResult(chunk=chunk, distance=dist, score=score)
                )
            return search_results

        return await asyncio.to_thread(_sync_search)

    async def delete_by_source(self, collection_name: str, source: str) -> int:
        """Delete all chunks from a given source (used on re-ingestion)."""

        def _sync_delete() -> int:
            col = self._get_collection(collection_name)
            results = col.get(where={"source": source})
            ids = results.get("ids", [])
            if ids:
                col.delete(ids=ids)
            return len(ids)

        return await asyncio.to_thread(_sync_delete)

    async def count(self, collection_name: str) -> int:
        def _sync_count() -> int:
            return self._get_collection(collection_name).count()

        return await asyncio.to_thread(_sync_count)

