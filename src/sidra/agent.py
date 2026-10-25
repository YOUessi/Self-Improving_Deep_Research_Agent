"""Auditable agent state machine and tool access control."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from sidra.retrieval import BM25Index
from sidra.schema import Action, AgentTrace, Document, Question


class Policy(Protocol):
    name: str

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action: ...


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_tool_calls: int = 7
    search_top_k: int = 3
    max_read_chars: int = 3500

    def __post_init__(self) -> None:
        if min(self.max_steps, self.max_tool_calls, self.search_top_k, self.max_read_chars) < 1:
            raise ValueError("Agent limits must be positive")


class SearchAgent:
    def __init__(self, docs: list[Document], policy: Policy, config: AgentConfig | None = None):
        self.docs = {d.doc_id: d for d in docs}
        self.index = BM25Index(docs)
        self.policy = policy
        self.config = config or AgentConfig()

    def run(self, question: Question) -> AgentTrace:
        config = self.config
        trace = AgentTrace(question_id=question.question_id, question=question.question, policy=self.policy.name)
        observations: list[dict] = []
        seen_hits: set[str] = set()
        read_ids: set[str] = set()
        begin = time.perf_counter()
        for step in range(config.max_steps):
            remaining = config.max_steps - step
            try:
                # Important: policy never receives question.gold_answer or supporting_doc_ids.
                action = self.policy.decide(question.question, observations, remaining)
                if not isinstance(action, Action):
                    raise TypeError("Policy must return an Action")
                entry = {"step": step, "action": action.to_dict()}
                if action.kind == "finish":
                    unknown = set(action.citations) - read_ids
                    if unknown:
                        raise ValueError(f"Unverified citations: {sorted(unknown)}; must read before cite")
                    trace.answer = action.answer.strip()
                    trace.citations = list(dict.fromkeys(action.citations))
                    trace.termination = "finish" if trace.answer else "abstained"
                    trace.steps.append(entry)
                    break
                if trace.tool_calls >= config.max_tool_calls:
                    trace.termination = "tool_budget_exhausted"
                    trace.steps.append({**entry, "error": "tool budget exhausted"})
                    break
                if action.kind == "search":
                    if not action.query.strip():
                        raise ValueError("Search query must be non-empty")
                    hits = self.index.search(action.query, k=config.search_top_k, scope=question.scope)
                    seen_hits.update(hit.doc_id for hit in hits)
                    result = {"kind": "search", "query": action.query, "hits": [h.as_dict() for h in hits]}
                    trace.search_calls += 1
                elif action.kind == "read":
                    if action.doc_id not in seen_hits:
                        raise ValueError("read requires a doc_id returned by an earlier search")
                    doc = self.docs[action.doc_id]
                    if doc.scope not in {question.scope, "global"}:
                        raise ValueError("read attempted outside question scope")
                    read_ids.add(doc.doc_id)
                    result = {"kind": "read", "doc_id": doc.doc_id, "title": doc.title,
                              "text": doc.text[:config.max_read_chars],
                              "truncated": len(doc.text) > config.max_read_chars}
                    trace.read_calls += 1
                else:
                    raise ValueError(f"Unsupported action: {action.kind}")
                trace.tool_calls += 1
                entry["observation"] = result
                observations.append(result)
                trace.steps.append(entry)
            except Exception as exc:
                # Store error class and a bounded explanation, not API keys or raw remote responses.
                trace.termination = "error"
                trace.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                trace.steps.append({"step": step, "error": trace.error})
                break
        if not trace.termination:
            trace.termination = "step_budget_exhausted"
        trace.elapsed_ms = round((time.perf_counter() - begin) * 1000)
        return trace
