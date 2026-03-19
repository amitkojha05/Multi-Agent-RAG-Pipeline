"""agents/critic_agent.py

CriticAgent acts as a fact-checker over the draft answer.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic

from agents.contextual_summariser import ScoredSummary


CRITIC_PROMPT = """You are a scientific fact-checker. Your job is to verify
whether the claims in an answer are supported by the provided source summaries.

Query: {query}

Source summaries (ground truth):
{sources}

Draft answer to verify:
{answer}

Respond with JSON only:
{{
  "confidence": <float 0.0-1.0>,
  "unsupported_claims": [<list of short strings describing claims with no source support>],
  "reasoning": "<one sentence explanation>"
}}"""


@dataclass
class CriticResult:
    confidence: float
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""
    latency_ms: float = 0.0
    flagged: bool = False


class CriticAgent:
    def __init__(self, model: Optional[str] = None) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.threshold = float(os.getenv("CRITIC_CONFIDENCE_THRESHOLD", "0.6"))

    async def critique(
        self,
        query: str,
        summaries: List[ScoredSummary],
        answer: str,
    ) -> CriticResult:
        t0 = time.perf_counter()

        sources_text = "\n".join(f"- [{s.source}]: {s.summary}" for s in summaries)
        prompt = CRITIC_PROMPT.format(
            query=query,
            sources=sources_text,
            answer=answer,
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(response.content[0].text.strip())
            confidence = float(data.get("confidence", 0.5))
            unsupported = data.get("unsupported_claims", [])
            reasoning = data.get("reasoning", "")
            return CriticResult(
                confidence=confidence,
                unsupported_claims=unsupported,
                reasoning=reasoning,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                flagged=confidence < self.threshold,
            )
        except Exception as e:  # pragma: no cover - error path
            return CriticResult(
                confidence=0.5,
                reasoning=f"Critic error: {e}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                flagged=False,
            )

