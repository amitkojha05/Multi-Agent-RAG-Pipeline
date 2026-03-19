"""store/pg_store.py

PGStore wraps asyncpg. Manages a connection pool and provides
execute/fetch helpers. The SQL agent builds queries against tables
created during CSV ingestion.
"""
import os
from typing import Any, List, Optional, Dict

import asyncpg


class PGStore:
    """
    Async PostgreSQL wrapper.

    Usage:
        store = PGStore()
        await store.connect()
        rows = await store.fetch("SELECT * FROM climate_data WHERE year > $1", 2020)
        await store.close()
    """

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "scirag"),
            user=os.getenv("POSTGRES_USER", "scirag_user"),
            password=os.getenv("POSTGRES_PASSWORD", "scirag_pass"),
            min_size=int(os.getenv("POSTGRES_MIN_POOL", "2")),
            max_size=int(os.getenv("POSTGRES_MAX_POOL", "10")),
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def execute(self, query: str, *args: Any) -> str:
        if not self._pool:
            raise RuntimeError("PGStore not connected")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("PGStore not connected")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchval(self, query: str, *args: Any) -> Any:
        if not self._pool:
            raise RuntimeError("PGStore not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def table_exists(self, table_name: str) -> bool:
        result = await self.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_name = $1)",
            table_name,
        )
        return bool(result)

    async def list_tables(self) -> List[str]:
        rows = await self.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        return [r["table_name"] for r in rows]

    async def describe_table(self, table_name: str) -> List[Dict[str, Any]]:
        """Returns column names + types — used by SQLAgent for query planning."""
        return await self.fetch(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = $1 ORDER BY ordinal_position",
            table_name,
        )

