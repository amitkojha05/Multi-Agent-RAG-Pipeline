"""ingestion/pdf_ingestor.py

PDFIngestor handles the full pipeline:
  PDF file → extract text per page → sliding-window chunks →
  ChromaDB upsert → entity extraction → GraphBuilder

Chunking strategy: 512 tokens, 64-token overlap.
Token counting is approximate (split on whitespace).
"""
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pdfplumber
import structlog

from store.chroma_store import Chunk, ChromaStore
from ingestion.graph_builder import GraphBuilder

log = structlog.get_logger()


class PDFIngestor:
    """
    Ingests PDF files into ChromaDB and builds entity graph.
    """

    def __init__(
        self,
        store: ChromaStore,
        graph_builder: GraphBuilder,
        collection: str = "papers",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> None:
        self.store = store
        self.graph_builder = graph_builder
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def ingest_file(self, pdf_path: str | Path) -> Dict[str, Any]:
        """
        Ingest a single PDF file.

        Returns:
            dict with keys: source, pages, chunks, entities
        """
        pdf_path = Path(pdf_path)
        source = pdf_path.name
        log.info("ingesting_pdf", source=source)

        # 1. Delete existing chunks for this source (idempotent re-ingestion)
        deleted = await self.store.delete_by_source(self.collection, source)
        if deleted > 0:
            log.info("deleted_existing_chunks", source=source, count=deleted)

        # 2. Extract text per page
        pages = self._extract_pages(pdf_path)

        # 3. Chunk pages
        chunks = self._chunk_pages(pages, source)

        # 4. Upsert to ChromaDB
        await self.store.upsert_chunks(self.collection, chunks)

        # 5. Extract entities and add to graph
        full_text = " ".join(p[1] for p in pages)
        entities = self.graph_builder.extract_and_add(full_text, source)

        log.info(
            "ingested_pdf",
            source=source,
            pages=len(pages),
            chunks=len(chunks),
            entities=len(entities),
        )
        return {
            "source": source,
            "pages": len(pages),
            "chunks": len(chunks),
            "entities": len(entities),
        }

    async def ingest_directory(self, directory: str | Path) -> List[Dict[str, Any]]:
        """Ingest all PDFs in a directory."""
        directory = Path(directory)
        results: List[Dict[str, Any]] = []
        for pdf_file in sorted(directory.glob("*.pdf")):
            result = await self.ingest_file(pdf_file)
            results.append(result)
        return results

    # ─── private helpers ──────────────────────────────────────

    def _extract_pages(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Returns list of (page_number, text) tuples.
        page_number is 1-indexed.
        Skips pages with fewer than 20 characters (blank/image-only pages).
        """
        pages: List[Tuple[int, str]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) >= 20:
                    pages.append((i, text))
        return pages

    def _chunk_pages(self, pages: List[Tuple[int, str]], source: str) -> List[Chunk]:
        """
        Sliding-window chunker over all pages.
        """
        token_page_pairs: List[Tuple[str, int]] = []
        for page_num, text in pages:
            for token in text.split():
                token_page_pairs.append((token, page_num))

        chunks: List[Chunk] = []
        i = 0
        chunk_index = 0
        while i < len(token_page_pairs):
            window = token_page_pairs[i : i + self.chunk_size]
            if not window:
                break
            tokens = [tp[0] for tp in window]
            page_num = window[0][1]
            text = " ".join(tokens)

            chunk_id = hashlib.sha256(
                f"{source}:{page_num}:{chunk_index}".encode()
            ).hexdigest()[:24]

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                    page=page_num,
                    chunk_index=chunk_index,
                    extra_meta={},
                )
            )
            chunk_index += 1
            i += self.chunk_size - self.overlap

        return chunks

