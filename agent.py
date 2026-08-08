"""A minimal but real memory-backed agent.

The loop is: recall() -> prompt -> Bedrock -> answer -> write back what was
learned.  Every turn records the retrieval id that fed it, so a bad answer is
traceable to the retrieval that caused it.  That link is the point of the
project: memory failure becoming agent failure, with the evidence to prove it.

    from agent import MemoryAgent
    agent = MemoryAgent("demo-agent-01", "session-1")
    turn = agent.answer("which database are we running in production?")
    print(turn.response, turn.memories_used, turn.retrieval_id)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import boto3
import psycopg

from memory import AWS_REGION, connect, recall, write_memory

# The account this project runs in has Bedrock access to Sonnet 4.5 and
# Haiku 4.5 only -- the newer Opus/Sonnet tiers return AccessDeniedException.
# Override with AGENT_MODEL_ID once model access is granted.
DEFAULT_MODEL_ID = os.getenv(
    "AGENT_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
# Fact extraction is a small, mechanical job -- run it on the cheap model.
EXTRACTOR_MODEL_ID = os.getenv(
    "AGENT_EXTRACTOR_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

SYSTEM_PROMPT = """\
You are an operations assistant for a small engineering team. You have a \
persistent memory; the memories retrieved for this question are the only \
record you hold of this deployment.

- When memories are provided, answer from them. Be specific, and prefer the \
memory's own wording for facts like versions, times, and names.
- When NO memories are provided, say so plainly in your first sentence -- \
that you have nothing in memory about this -- and then give only what general \
knowledge allows. Never invent specifics about this deployment: no versions, \
no hostnames, no dates, no names.
- Two or three sentences. No preamble.\
"""

EXTRACTOR_PROMPT = """\
Extract durable facts stated by the USER that are worth remembering for future \
conversations -- things about their systems, decisions, or preferences.

Return one fact per line, rewritten as a standalone declarative sentence. \
Return at most 2. If the user stated no durable fact (they only asked a \
question), return exactly: NONE\
"""

_bedrock = None


def _client():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock


@dataclass
class AgentTurn:
    """One agent exchange, plus the retrieval telemetry that produced it."""

    turn_id: UUID | None
    agent_id: str
    session_id: str
    query: str
    response: str
    memories_used: int
    retrieval_id: UUID | None
    raw_candidates: int | None
    top_similarity: float | None
    applied_floor: float | None
    model_id: str
    latency_ms: int
    memories: list[dict[str, Any]] = field(default_factory=list)
    remembered: list[str] = field(default_factory=list)
    created_at: datetime | None = None

    @property
    def grounded(self) -> bool:
        """Did any memory actually reach the model?"""
        return self.memories_used > 0

    @property
    def was_near_miss(self) -> bool:
        """Did the floor discard something that was probably relevant?"""
        if self.memories_used or not self.raw_candidates or self.top_similarity is None:
            return False
        if self.applied_floor is None:
            return False
        return self.applied_floor - 0.15 <= self.top_similarity < self.applied_floor


class ModelAccessError(RuntimeError):
    """Bedrock refused the model for account reasons, not request reasons."""


def _converse(model_id: str, system: str, user: str, max_tokens: int = 512) -> str:
    try:
        response = _client().converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
    except _client().exceptions.ResourceNotFoundException as exc:
        raise ModelAccessError(
            f"Bedrock will not serve {model_id}: {exc}\n\n"
            "This is an account entitlement, not a bug in this code. Submit the "
            "Anthropic use case details form in the Bedrock console "
            "(Model access -> Anthropic -> submit use case details), then retry. "
            "Titan embeddings are unaffected, so memory.py and checks.py keep working."
        ) from exc
    except _client().exceptions.AccessDeniedException as exc:
        raise ModelAccessError(
            f"Bedrock denied {model_id}: {exc}\n\n"
            "Request access to this model in the Bedrock console, or set "
            "AGENT_MODEL_ID to a model this account is entitled to."
        ) from exc
    return "".join(
        block.get("text", "") for block in response["output"]["message"]["content"]
    ).strip()


def _build_prompt(query: str, memories: list[dict[str, Any]]) -> str:
    if not memories:
        return (
            "<memories>\n(no memories were retrieved for this question)\n</memories>\n\n"
            f"Question: {query}"
        )
    lines = [
        f"- [{m['kind']}, similarity {m['similarity']:.2f}] {m['content']}"
        for m in memories
    ]
    return "<memories>\n" + "\n".join(lines) + f"\n</memories>\n\nQuestion: {query}"


class MemoryAgent:
    """Recall -> reason -> remember, with the retrieval link recorded."""

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        extractor_model_id: str = EXTRACTOR_MODEL_ID,
        limit: int = 5,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.model_id = model_id
        self.extractor_model_id = extractor_model_id
        self.limit = limit

    def answer(
        self,
        query: str,
        *,
        remember: bool = True,
        conn: psycopg.Connection | None = None,
    ) -> AgentTurn:
        """Answer `query` from memory and record the turn.

        `remember` controls the write-back step.  Turn it off when comparing the
        same question across conditions -- otherwise the agent's own writes
        change the corpus between measurements.
        """

        def _run(c: psycopg.Connection) -> AgentTurn:
            started = time.perf_counter()
            stats: dict[str, Any] = {}
            memories = recall(
                self.agent_id,
                query,
                session_id=self.session_id,
                limit=self.limit,
                stats=stats,
                conn=c,
            )
            response = _converse(
                self.model_id, SYSTEM_PROMPT, _build_prompt(query, memories)
            )
            latency_ms = int((time.perf_counter() - started) * 1000)

            remembered: list[str] = []
            if remember:
                remembered = self._remember(query, response, conn=c)

            with c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_turns
                        (agent_id, session_id, query, response, retrieval_id,
                         memories_used, model_id, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        self.agent_id,
                        self.session_id,
                        query,
                        response,
                        stats.get("retrieval_id"),
                        len(memories),
                        self.model_id,
                        latency_ms,
                    ),
                )
                row = cur.fetchone()

            return AgentTurn(
                turn_id=row["id"],
                agent_id=self.agent_id,
                session_id=self.session_id,
                query=query,
                response=response,
                memories_used=len(memories),
                retrieval_id=stats.get("retrieval_id"),
                raw_candidates=stats.get("raw_candidates"),
                top_similarity=stats.get("top_similarity"),
                applied_floor=stats.get("applied_floor"),
                model_id=self.model_id,
                latency_ms=latency_ms,
                memories=memories,
                remembered=remembered,
                created_at=row["created_at"],
            )

        if conn is not None:
            return _run(conn)
        with connect() as owned:
            return _run(owned)

    def _remember(
        self, query: str, response: str, *, conn: psycopg.Connection
    ) -> list[str]:
        """Write back durable facts the user stated, so memory accumulates."""
        extracted = _converse(
            self.extractor_model_id,
            EXTRACTOR_PROMPT,
            f"User: {query}\n\nAssistant: {response}",
            max_tokens=200,
        )
        if not extracted or extracted.strip().upper().startswith("NONE"):
            return []

        facts = [
            line.strip().lstrip("-•* ").strip()
            for line in extracted.splitlines()
            if line.strip() and not line.strip().upper().startswith("NONE")
        ][:2]

        written = []
        for fact in facts:
            write_memory(
                self.agent_id,
                self.session_id,
                "fact",
                fact,
                metadata={"source": "agent_writeback", "from_query": query},
                conn=conn,
            )
            written.append(fact)
        return written


def _demo() -> None:
    import sys

    agent_id = sys.argv[1] if len(sys.argv) > 1 else "demo-agent-01"
    query = (
        " ".join(sys.argv[2:])
        if len(sys.argv) > 2
        else "which database are we running in production?"
    )
    agent = MemoryAgent(agent_id, "cli-session")
    turn = agent.answer(query, remember=False)
    print(f"query      {turn.query}")
    print(f"model      {turn.model_id}")
    print(f"memories   {turn.memories_used} used of {turn.raw_candidates} candidates")
    print(f"retrieval  {turn.retrieval_id}")
    print(f"top sim    {turn.top_similarity}  floor {turn.applied_floor}")
    print(f"grounded   {turn.grounded}   near-miss {turn.was_near_miss}")
    print(f"latency    {turn.latency_ms} ms")
    print(f"\n{turn.response}")


if __name__ == "__main__":
    _demo()
