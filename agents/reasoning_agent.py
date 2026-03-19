"""agents/reasoning_agent.py

ReasoningAgent synthesises an answer from scored context summaries.
It exposes calculator and code_executor tools to Claude.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic
import structlog

from agents.contextual_summariser import ScoredSummary
from tools.calculator import safe_eval
from tools.code_executor import CodeExecutor

log = structlog.get_logger()


TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression safely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python math expression, e.g. 6.022e23 * 1.38e-23",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "code_executor",
        "description": "Execute a short Python snippet for data analysis. Returns stdout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Print results to stdout.",
                }
            },
            "required": ["code"],
        },
    },
]


SYSTEM_PROMPT = """You are a scientific research assistant. Answer the user's research
question using ONLY the provided context summaries.

Rules:
- Cite sources inline using the format [source: filename, page N]
- If a claim comes from a structured dataset, cite as [source: table_name]
- Do NOT make claims not supported by the context
- If the context is insufficient, say so explicitly
- You may use the calculator or code_executor tools for quantitative reasoning
- Keep the answer focused and concise (100-300 words unless the question demands more)"""


@dataclass
class ReasoningResult:
    answer: str
    sources_used: List[str]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class ReasoningAgent:
    """Tool-using reasoning agent over scored summaries."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self._code_executor = CodeExecutor()

    async def reason(
        self, query: str, summaries: List[ScoredSummary]
    ) -> ReasoningResult:
        t0 = time.perf_counter()

        if not summaries:
            return ReasoningResult(
                answer="Insufficient context to answer this question.",
                sources_used=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        context_block = self._build_context(summaries)
        user_message = f"{context_block}\n\nQuestion: {query}"

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message}]
        tool_calls_log: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_input_tokens += getattr(usage, "input_tokens", 0)
                total_output_tokens += getattr(usage, "output_tokens", 0)

            stop_reason = getattr(response, "stop_reason", None)

            if stop_reason == "end_turn":
                answer = " ".join(
                    block.text for block in response.content if block.type == "text"
                )
                break

            if stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results_blocks: List[Dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = self._execute_tool(block.name, block.input)
                    tool_calls_log.append(
                        {"tool": block.name, "input": block.input, "result": result}
                    )
                    tool_results_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )
                messages.append({"role": "user", "content": tool_results_blocks})
                continue

            answer = "Unable to generate answer."
            break

        sources_used = list({s.source for s in summaries})

        return ReasoningResult(
            answer=answer,
            sources_used=sources_used,
            tool_calls=tool_calls_log,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def _build_context(self, summaries: List[ScoredSummary]) -> str:
        lines = ["## Retrieved Context", ""]
        for i, s in enumerate(summaries, start=1):
            page_str = f", page {s.page}" if s.page else ""
            lines.append(
                f"[{i}] Source: {s.source}{page_str} (relevance: {s.relevance_score}/5)"
            )
            lines.append(s.summary)
            lines.append("")
        return "\n".join(lines)

    def _execute_tool(self, name: str, inputs: dict) -> str:
        if name == "calculator":
            return safe_eval(inputs.get("expression", ""))
        if name == "code_executor":
            return self._code_executor.execute(inputs.get("code", ""))
        return f"Unknown tool: {name}"

