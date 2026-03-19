"""ingestion/csv_ingestor.py

CSVIngestor ingests a CSV file into:
  1. PostgreSQL — for structured SQL queries
  2. ChromaDB "datasets" collection — for semantic search over row content
  3. GraphBuilder — entity extraction from text columns

Table name is derived from the CSV filename (lowercased, spaces→underscores).
"""
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import structlog

from ingestion.graph_builder import GraphBuilder
from store.chroma_store import Chunk, ChromaStore
from store.pg_store import PGStore

log = structlog.get_logger()


class CSVIngestor:
    def __init__(
        self,
        chroma_store: ChromaStore,
        pg_store: PGStore,
        graph_builder: GraphBuilder,
        dataset_collection: str = "datasets",
    ) -> None:
        self.chroma = chroma_store
        self.pg = pg_store
        self.graph_builder = graph_builder
        self.collection = dataset_collection

    async def ingest_file(self, csv_path: str | Path) -> Dict[str, Any]:
        csv_path = Path(csv_path)
        table_name = self._table_name(csv_path.stem)
        log.info("ingesting_csv", source=csv_path.name, table=table_name)

        df = pd.read_csv(csv_path)
        df = df.where(pd.notna(df), None)

        await self._create_table(table_name, df)
        await self._insert_rows(table_name, df)

        chunks = self._rows_to_chunks(df, csv_path.name)
        await self.chroma.delete_by_source(self.collection, csv_path.name)
        await self.chroma.upsert_chunks(self.collection, chunks)

        text_cols = df.select_dtypes(include="object").columns.tolist()
        combined_text = " ".join(
            " ".join(str(v) for v in df[col].dropna()) for col in text_cols
        )
        entities = self.graph_builder.extract_and_add(combined_text, csv_path.name)

        log.info(
            "ingested_csv",
            table=table_name,
            rows=len(df),
            chunks=len(chunks),
            entities=len(entities),
        )
        return {
            "table": table_name,
            "rows": len(df),
            "chunks": len(chunks),
            "entities": len(entities),
        }

    @staticmethod
    def _table_name(stem: str) -> str:
        name = re.sub(r"[^a-z0-9_]", "_", stem.lower())
        return name[:63]

    async def _create_table(self, table_name: str, df: pd.DataFrame) -> None:
        type_map = {
            "int64": "BIGINT",
            "float64": "DOUBLE PRECISION",
            "bool": "BOOLEAN",
            "object": "TEXT",
        }
        cols: List[str] = []
        for col, dtype in df.dtypes.items():
            pg_type = type_map.get(str(dtype), "TEXT")
            safe_col = re.sub(r"[^a-z0-9_]", "_", col.lower())
            cols.append(f'"{safe_col}" {pg_type}')

        ddl = f"DROP TABLE IF EXISTS {table_name}; CREATE TABLE {table_name} ({', '.join(cols)})"
        await self.pg.execute(ddl)

    async def _insert_rows(self, table_name: str, df: pd.DataFrame) -> None:
        if not self.pg._pool:
            raise RuntimeError("PGStore not connected")
        cols = [re.sub(r"[^a-z0-9_]", "_", c.lower()) for c in df.columns]
        records = [tuple(row) for row in df.itertuples(index=False, name=None)]
        async with self.pg._pool.acquire() as conn:  # type: ignore[attr-defined]
            await conn.copy_records_to_table(
                table_name, records=records, columns=cols
            )

    def _rows_to_chunks(self, df: pd.DataFrame, source: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        for i, row in df.iterrows():
            text = " | ".join(
                f"{k}: {v}" for k, v in row.items() if v is not None
            )
            chunk_id = hashlib.sha256(
                f"{source}:row:{i}".encode()
            ).hexdigest()[:24]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                    page=None,
                    chunk_index=int(i),
                    extra_meta={"row_index": int(i)},
                )
            )
        return chunks

