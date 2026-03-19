"""agents/sql_agent.py

SQLAgent translates natural language queries into safe SELECT
statements against PostgreSQL. Uses Claude to generate SQL.
Refuses to execute any non-SELECT statement.
"""
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import anthropic
import structlog

from store.pg_store import PGStore

log = structlog.get_logger()


@dataclass
class SQLResult:
    rows: List[dict]
    sql_used: str
    table_searched: Optional[str]
    latency_ms: float
    error: Optional[str] = None


class SQLAgent:
    """Natural-language → SQL → PostgreSQL specialist."""

    def __init__(self, pg_store: PGStore, model: Optional[str] = None) -> None:
        self.pg = pg_store
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    async def search(self, query: str) -> SQLResult:
        t0 = time.perf_counter()

        tables = await self.pg.list_tables()
        if not tables:
            return SQLResult(
                rows=[],
                sql_used="",
                table_searched=None,
                latency_ms=0.0,
                error="No tables available",
            )

        schema_context = await self._build_schema_context(tables)
        sql = await self._generate_sql(query, schema_context)

        if not sql:
            return SQLResult(
                rows=[],
                sql_used="",
                table_searched=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="Could not generate SQL for this query",
            )

        if not self._is_safe_select(sql):
            log.warning("unsafe_sql_rejected", sql=sql)
            return SQLResult(
                rows=[],
                sql_used=sql,
                table_searched=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="Only SELECT queries are permitted",
            )

        try:
            rows = await self.pg.fetch(sql)
        except Exception as e:  # pragma: no cover - DB errors integration-tested
            return SQLResult(
                rows=[],
                sql_used=sql,
                table_searched=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(e),
            )

        return SQLResult(
            rows=rows[:50],
            sql_used=sql,
            table_searched=tables[0] if tables else None,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    async def _build_schema_context(self, tables: List[str]) -> str:
        lines = ["Available PostgreSQL tables:"]
        for table in tables:
            cols = await self.pg.describe_table(table)
            col_str = ", ".join(
                f"{c['column_name']} ({c['data_type']})" for c in cols
            )
            lines.append(f"  {table}: {col_str}")
        return "\n".join(lines)

    async def _generate_sql(self, query: str, schema: str) -> Optional[str]:
        prompt = f"""You are a SQL expert. Given a natural language question and a
database schema, write a single PostgreSQL SELECT query that answers the question.

Rules:
- Write ONLY the SQL statement, nothing else
- Only use SELECT (no INSERT, UPDATE, DELETE, DROP)
- Use LIMIT 50 to avoid large result sets
- If the question cannot be answered with SQL, respond with: CANNOT_ANSWER

Schema:
{schema}

Question: {query}

SQL:"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        sql = response.content[0].text.strip()
        return None if sql == "CANNOT_ANSWER" else sql

    @staticmethod
    def _is_safe_select(sql: str) -> bool:
        upper = sql.strip().upper()
        if not upper.startswith("SELECT"):
            return False
        dangerous = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "TRUNCATE",
            "ALTER",
            "CREATE",
            "GRANT",
            "EXEC",
            "--",
            ";--",
        ]
        return not any(kw in upper for kw in dangerous)

