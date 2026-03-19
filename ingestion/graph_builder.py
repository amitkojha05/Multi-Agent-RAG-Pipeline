"""ingestion/graph_builder.py

GraphBuilder extracts named entities using spaCy NER and builds
a NetworkX graph of entity co-occurrences.

Graph semantics:
  - Nodes: entity strings (lowercased)
  - Node attributes: {"type": spaCy entity label, "sources": set of filenames}
  - Edges: co-occurrence in same source document
  - Edge attributes: {"weight": co-occurrence count, "sources": set of filenames}

The graph is persisted to graph.pkl and loaded on startup.
"""
import os
import pickle
from pathlib import Path
from typing import List, Set

import networkx as nx
import spacy
import structlog

log = structlog.get_logger()
GRAPH_PATH = Path(os.getenv("GRAPH_PERSIST_PATH", "./graph.pkl"))


class GraphBuilder:
    """Builds and persists an entity co-occurrence graph."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._nlp = spacy.load(spacy_model)
        self._graph = self._load_or_create()

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def extract_and_add(self, text: str, source: str) -> List[str]:
        """Extract entities from text, add to graph, persist, and return list."""
        doc = self._nlp(text[:100_000])  # avoid excessive memory use
        entities: List[str] = []
        for ent in doc.ents:
            if ent.label_ in {
                "CARDINAL",
                "ORDINAL",
                "PERCENT",
                "MONEY",
                "TIME",
                "DATE",
                "QUANTITY",
            }:
                continue
            entity = ent.text.strip().lower()
            if len(entity) < 2:
                continue
            entities.append(entity)

            if not self._graph.has_node(entity):
                self._graph.add_node(entity, type=ent.label_, sources=set())
            self._graph.nodes[entity]["sources"].add(source)

        # Add co-occurrence edges between all entity pairs in this source
        seen: Set[str] = set(entities)
        ents_list = list(seen)
        for i, e1 in enumerate(ents_list):
            for e2 in ents_list[i + 1 :]:
                if e1 == e2:
                    continue
                if self._graph.has_edge(e1, e2):
                    self._graph[e1][e2]["weight"] += 1
                    self._graph[e1][e2]["sources"].add(source)
                else:
                    self._graph.add_edge(e1, e2, weight=1, sources={source})

        self._persist()
        return list(seen)

    def related_entities(
        self,
        entity: str,
        max_hops: int = 2,
        min_weight: int = 1,
    ) -> List[str]:
        """Return entities within max_hops reachable via edges with weight >= min_weight."""
        entity = entity.lower()
        if not self._graph.has_node(entity):
            return []

        filtered = nx.Graph(
            (u, v, d)
            for u, v, d in self._graph.edges(data=True)
            if d.get("weight", 0) >= min_weight
        )
        if not filtered.has_node(entity):
            return []

        reachable = nx.single_source_shortest_path_length(
            filtered, entity, cutoff=max_hops
        )
        return [
            n for n, _ in sorted(reachable.items(), key=lambda x: x[1]) if n != entity
        ]

    def sources_for_entity(self, entity: str) -> Set[str]:
        """Return all source files mentioning this entity."""
        entity = entity.lower()
        if not self._graph.has_node(entity):
            return set()
        return self._graph.nodes[entity].get("sources", set())

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ─── private ──────────────────────────────────────────────

    def _load_or_create(self) -> nx.Graph:
        if GRAPH_PATH.exists():
            log.info("loading_graph", path=str(GRAPH_PATH))
            with open(GRAPH_PATH, "rb") as f:
                return pickle.load(f)
        log.info("creating_empty_graph")
        return nx.Graph()

    def _persist(self) -> None:
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self._graph, f)

