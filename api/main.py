"""api/main.py — FastAPI application entry point."""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    QueryRequest,
    QueryResponse as QueryResponseModel,
    IngestRequest,
    IngestResponse,
    HealthResponse,
)
from agents.orchestrator import Orchestrator, QueryResponse
from agents.vector_agent import VectorAgent
from agents.sql_agent import SQLAgent
from agents.graph_agent import GraphAgent
from agents.contextual_summariser import ContextualSummariser
from agents.reasoning_agent import ReasoningAgent
from agents.critic_agent import CriticAgent
from cache.redis_cache import RedisCache
from store.chroma_store import ChromaStore
from store.pg_store import PGStore
from ingestion.pdf_ingestor import PDFIngestor
from ingestion.csv_ingestor import CSVIngestor
from ingestion.graph_builder import GraphBuilder

log = structlog.get_logger()

_chroma: ChromaStore | None = None
_pg: PGStore | None = None
_cache: RedisCache | None = None
_graph_builder: GraphBuilder | None = None
_orchestrator: Orchestrator | None = None
_pdf_ingestor: PDFIngestor | None = None
_csv_ingestor: CSVIngestor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chroma, _pg, _cache, _graph_builder
    global _orchestrator, _pdf_ingestor, _csv_ingestor

    _chroma = ChromaStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    _pg = PGStore()
    await _pg.connect()

    _cache = RedisCache()
    await _cache.connect()

    _graph_builder = GraphBuilder()
    asyncio.create_task(_cache.subscribe_invalidation())

    vector_agent = VectorAgent(_chroma)
    sql_agent = SQLAgent(_pg)
    graph_agent = GraphAgent(_graph_builder)
    summariser = ContextualSummariser()
    reasoner = ReasoningAgent()
    critic = CriticAgent()

    _orchestrator = Orchestrator(
        vector_agent=vector_agent,
        sql_agent=sql_agent,
        graph_agent=graph_agent,
        summariser=summariser,
        reasoning_agent=reasoner,
        critic_agent=critic,
        cache=_cache,
    )

    _pdf_ingestor = PDFIngestor(_chroma, _graph_builder)
    _csv_ingestor = CSVIngestor(_chroma, _pg, _graph_builder)

    log.info("startup_complete")
    yield

    await _pg.close()
    await _cache.close()
    log.info("shutdown_complete")


app = FastAPI(
    title="Multi-Agent RAG Pipeline",
    description="Scientific research question answering over heterogeneous sources",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequest):
    if not _orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialised")
    try:
        result: QueryResponse = await _orchestrator.query(request.query)
        return result
    except Exception as e:  # pragma: no cover - integration path
        log.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    if not (_pdf_ingestor and _csv_ingestor and _cache):
        raise HTTPException(status_code=500, detail="Ingestors not initialised")

    path = Path(request.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        if request.source_type == "pdf":
            details = await _pdf_ingestor.ingest_file(path)
        else:
            details = await _csv_ingestor.ingest_file(path)

        await _cache.publish_invalidation(path.name)

        return IngestResponse(status="success", source=path.name, details=details)
    except Exception as e:  # pragma: no cover - integration path
        log.error("ingest_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    if not (_chroma and _pg and _cache and _graph_builder):
        raise HTTPException(status_code=500, detail="Components not initialised")

    chroma_docs = await _chroma.count("papers") + await _chroma.count("datasets")
    pg_tables = await _pg.list_tables()
    try:
        await _cache._redis.ping()  # type: ignore[attr-defined]
        cache_ok = True
    except Exception:
        cache_ok = False

    return HealthResponse(
        status="ok",
        chroma_docs=chroma_docs,
        pg_tables=pg_tables,
        graph_nodes=_graph_builder.node_count(),
        graph_edges=_graph_builder.edge_count(),
        cache_connected=cache_ok,
    )


@app.get("/graph/{entity}")
async def graph_neighbours(entity: str, hops: int = 2):
    if not _graph_builder:
        raise HTTPException(status_code=500, detail="Graph builder not initialised")
    related = _graph_builder.related_entities(entity, max_hops=hops)
    sources = list(_graph_builder.sources_for_entity(entity))
    return {
        "entity": entity,
        "related": related[:20],
        "sources": sources,
        "hops": hops,
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )

