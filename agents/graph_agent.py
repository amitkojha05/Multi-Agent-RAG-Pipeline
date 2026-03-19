"""agents/graph_agent.py

GraphAgent extracts entity mentions from the query,
traverses the entity co-occurrence graph, and returns
related entities and their source documents.
"""
import time
from dataclasses import dataclass
from typing import List, Set

import spacy

from ingestion.graph_builder import GraphBuilder


@dataclass
class GraphResult:
    query_entities: List[str]
    related_entities: List[str]
    source_documents: List[str]
    traversal_depth: int
    latency_ms: float


class GraphAgent:
    """Entity graph traversal specialist."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        max_hops: int = 2,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.graph_builder = graph_builder
        self.max_hops = max_hops
        self._nlp = spacy.load(spacy_model)

    async def search(self, query: str) -> GraphResult:
        t0 = time.perf_counter()

        query_entities = self._extract_entities(query)
        if not query_entities:
            return GraphResult(
                query_entities=[],
                related_entities=[],
                source_documents=[],
                traversal_depth=0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        all_related: Set[str] = set()
        all_sources: Set[str] = set()
        for entity in query_entities:
            related = self.graph_builder.related_entities(entity, self.max_hops)
            all_related.update(related)
            for r in related:
                all_sources.update(self.graph_builder.sources_for_entity(r))

        all_related -= set(query_entities)

        return GraphResult(
            query_entities=query_entities,
            related_entities=sorted(all_related)[:30],
            source_documents=sorted(all_sources),
            traversal_depth=self.max_hops,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _extract_entities(self, text: str) -> List[str]:
        doc = self._nlp(text)
        entities: List[str] = []
        skip_labels = {
            "CARDINAL",
            "ORDINAL",
            "PERCENT",
            "MONEY",
            "TIME",
            "DATE",
            "QUANTITY",
        }
        for ent in doc.ents:
            if ent.label_ in skip_labels:
                continue
            entity = ent.text.strip().lower()
            if len(entity) >= 2 and entity not in entities:
                entities.append(entity)
        return entities

