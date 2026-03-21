# Multi-Agent RAG Pipeline over Scientific Data

## Overview
This project implements a production-style multi-agent retrieval and reasoning pipeline for scientific question answering over heterogeneous data sources. It combines semantic retrieval over PDF/CSV embeddings, structured SQL querying, and entity-graph traversal, then performs context filtering, reasoning with tools, and hallucination checking before returning an answer with citations and latency breakdown.

## Features Implemented
- **Level 1 (Core)**: PDF + CSV ingestion, `VectorAgent` / `SQLAgent` / `GraphAgent`, contextual summariser, `ReasoningAgent` with `calculator` + `code_executor`, FastAPI `/query` endpoint returning answer + sources + latency.
- **Level 2 (Scalability)**: `asyncio.gather` parallel retrieval, Redis TTL cache with ingestion-triggered invalidation via pub/sub, NetworkX entity graph traversal.
- **Level 3 (Robustness)**: `CriticAgent` (LLM-as-judge), subprocess-based code sandbox, evaluation harness with keyword recall, source recall, confidence, flagged rate, and latency metrics.

## Quick Start
```bash
cp .env.example .env
# add your ANTHROPIC_API_KEY in .env

docker-compose up -d

pip install -r requirements.txt
python -m spacy download en_core_web_sm

uvicorn api.main:app --reload --port 8000
```

## Ingest Data
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"source_type\":\"pdf\",\"file_path\":\"data/sample_papers/enzyme_kinetics.pdf\"}"

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"source_type\":\"csv\",\"file_path\":\"data/sample_dataset.csv\"}"
```

## Query the System
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What are the effects of temperature on enzyme activity?\"}"
```

## Key Architectural Decisions

### No LangChain / LlamaIndex
All agents call the Anthropic SDK directly with explicit prompts and explicit control flow. This keeps behavior inspectable and auditable: each prompt, token decision, tool call, and response transformation is visible in plain Python rather than hidden behind framework abstractions.

### Contextual Summarisation (paper-qa inspired)
Retrieved chunks are scored (1-5 relevance) and summarized before reasoning. This reduces noisy retrieval context and improves grounding quality by forwarding only high-relevance evidence to the final answer synthesis stage.

### Parallel Specialist Agents
Vector, SQL, and graph retrieval run concurrently with `asyncio.gather`. This keeps end-to-end retrieval latency close to the slowest branch instead of the sum of all branches and improves throughput under multiple concurrent queries.

### NetworkX over Neo4j
For assignment-scale graphs, in-memory NetworkX offers low operational overhead and simple traversal logic. Persisting to `graph.pkl` retains graph state across restarts while keeping setup lightweight.

### Redis Cache Invalidation Strategy
A TTL cache accelerates repeated questions. On ingestion, the system publishes an invalidation event and flushes cached query responses to prevent stale answers. This coarse invalidation is simple and correct, with a clear path to finer-grained invalidation.

## API Endpoints
- `POST /query` - run full multi-agent pipeline
- `POST /ingest` - ingest PDF/CSV and trigger cache invalidation
- `GET /health` - service readiness and store counts
- `GET /graph/{entity}` - inspect graph neighbors

## Evaluation
Run:
```bash
python -m eval.harness
```
This reads `eval/qa_pairs.json`, queries the API, and writes aggregate and per-question results to `eval/results.json`.

## Known Limitations
- Coarse cache invalidation flushes all query keys on new ingestion.
- spaCy `en_core_web_sm` can miss domain-specific scientific entities.
- The code executor blocks common dangerous imports but is not a hardened sandbox.
- No response streaming; answers are returned after full pipeline completion.

## One Thing I'd Do Differently
I would replace heuristic retrieval blending with a trained routing/reranking stage that predicts which retrieval branch (vector, SQL, graph, or hybrid) should dominate for each query and calibrates evidence confidence before reasoning.

## Environment Variables
See `.env.example` for full configuration, including model, stores, cache, retrieval thresholds, critic threshold, and API settings.

