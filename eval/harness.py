"""eval/harness.py

Runs API-level QA evaluation over ground-truth pairs.
"""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx


@dataclass
class EvalResult:
    question_id: str
    question: str
    answer: str
    keyword_recall: float
    source_recall: float
    confidence: float
    flagged: bool
    latency_ms: int
    error: str = ""


class EvalHarness:
    def __init__(self, api_base: str = "http://localhost:8000") -> None:
        self.api_base = api_base

    async def run(
        self,
        qa_file: str = "eval/qa_pairs.json",
        output_file: str = "eval/results.json",
    ) -> dict:
        qa_pairs = json.loads(Path(qa_file).read_text(encoding="utf-8"))
        results: List[EvalResult] = []

        async with httpx.AsyncClient(timeout=60) as client:
            for qa in qa_pairs:
                result = await self._eval_one(client, qa)
                results.append(result)
                print(
                    f"[{qa['id']}] recall={result.keyword_recall:.2f} "
                    f"conf={result.confidence:.2f} latency={result.latency_ms}ms"
                )

        total = len(results) if results else 1
        metrics = {
            "total": len(results),
            "avg_keyword_recall": sum(r.keyword_recall for r in results) / total,
            "avg_source_recall": sum(r.source_recall for r in results) / total,
            "avg_confidence": sum(r.confidence for r in results) / total,
            "flagged_rate": sum(1 for r in results if r.flagged) / total,
            "avg_latency_ms": sum(r.latency_ms for r in results) / total,
            "errors": sum(1 for r in results if r.error),
        }

        output = {"metrics": metrics, "results": [r.__dict__ for r in results]}
        Path(output_file).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
        return metrics

    async def _eval_one(self, client: httpx.AsyncClient, qa: dict) -> EvalResult:
        try:
            resp = await client.post(
                f"{self.api_base}/query",
                json={"query": qa["question"]},
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = data.get("latency", {}).get("total_ms", 0)

            answer_lower = data.get("answer", "").lower()
            sources_cited = [s.get("source", "") for s in data.get("sources", [])]

            keywords = qa.get("expected_answer_keywords", [])
            keyword_recall = (
                sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords)
                if keywords
                else 1.0
            )

            exp_sources = qa.get("expected_sources", [])
            source_recall = (
                sum(1 for s in exp_sources if any(s in cited for cited in sources_cited))
                / len(exp_sources)
                if exp_sources
                else 1.0
            )

            return EvalResult(
                question_id=qa["id"],
                question=qa["question"],
                answer=data.get("answer", ""),
                keyword_recall=keyword_recall,
                source_recall=source_recall,
                confidence=float(data.get("confidence", 0.0)),
                flagged=bool(data.get("flagged", False)),
                latency_ms=int(latency_ms),
            )
        except Exception as e:
            return EvalResult(
                question_id=qa.get("id", ""),
                question=qa.get("question", ""),
                answer="",
                keyword_recall=0.0,
                source_recall=0.0,
                confidence=0.0,
                flagged=False,
                latency_ms=0,
                error=str(e),
            )


if __name__ == "__main__":
    harness = EvalHarness()
    asyncio.run(harness.run())

