"""Opt-in LangGraph runtime for evidence-grounded, multi-role research.

The model NEVER sees Question.gold_answer, supporting_doc_ids or evaluator state.
Search/READ use the existing scoped BM25 corpus. All accepted actions remain in
AgentTrace, so the existing TRAIN-only scorer and GRPO scripts can evaluate them.

Roles are explicit graph nodes, not new model checkpoints. "single" deliberately
uses the original Policy.decide contract and prompt for matched-policy GRPO.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from sidra.agent import AgentConfig, Policy
from sidra.evidence_audit import EvidenceAudit
from sidra.evidence_state import derive_evidence_state
from sidra.policies import SYSTEM
from sidra.retrieval import BM25Index
from sidra.schema import Action, AgentTrace, Document, Question


class ResearchState(TypedDict, total=False):
    question_id: str
    question: str
    scope: str
    observations: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    seen_hit_ids: list[str]
    read_ids: list[str]
    evidence_summary: dict[str, Any] | None
    steps_used: int
    tool_calls: int
    search_calls: int
    read_calls: int
    pending_action: dict[str, Any] | None
    answer: str
    citations: list[str]
    termination: str
    error: str


ROLE_GUIDANCE = {
    "planner": "Planner: identify the first missing fact. Propose a legal SEARCH or READ.",
    "researcher": (
        "Researcher: follow the missing hop by SEARCH/READ. Do not claim that an "
        "unread snippet proves the answer. Avoid unsupported FINISH actions."
    ),
    "solver": (
        "Solver: combine actually READ sources; answer only with supporting READ "
        "citations. Continue SEARCH/READ when evidence is inadequate."
    ),
}


class GraphResearchAgent:
    """LangGraph orchestration over existing SIDRA tools, policy and trace contract.

    Optional checkpointer is caller owned. This class does NOT automatically replay
    partially executed LLM calls; a durable checkpoint is not an exactly-once receipt.
    """

    def __init__(
        self,
        documents: list[Document],
        policy: Policy,
        config: AgentConfig | None = None,
        *,
        role_mode: Literal["multi", "single"] = "multi",
        role_policies: Mapping[str, Policy] | None = None,
        min_citations: int = 2,
        evidence_state_mode: Literal["off", "v1", "v2", "v3"] = "off",
        checkpointer: Any = None,
        thread_namespace: str = "sidra",
    ) -> None:
        if role_mode not in {"multi", "single"}:
            raise ValueError("role_mode must be multi or single")
        if evidence_state_mode not in {"off", "v1", "v2", "v3"}:
            raise ValueError("invalid evidence-state mode")
        if role_policies and role_mode != "multi":
            raise ValueError("separate role policies require multi role mode")
        if role_policies and not set(role_policies).issubset(ROLE_GUIDANCE):
            raise ValueError("unsupported role policy key")
        if min_citations < 1:
            raise ValueError("min_citations must be positive")
        if not thread_namespace:
            raise ValueError("thread_namespace required")
        self.docs = {d.doc_id: d for d in documents}
        self.index = BM25Index(documents, title_boost=(config or AgentConfig()).title_boost)
        self.config = config or AgentConfig()
        self.policy = policy
        self.role_policies = dict(role_policies or {})
        self.role_mode = role_mode
        self.min_citations = min_citations
        self.evidence_state_mode = evidence_state_mode
        self.thread_namespace = thread_namespace
        self.last_thread_id: str | None = None
        self.checkpointer = checkpointer
        graph = StateGraph(ResearchState)
        graph.add_node("observe", self._observe)
        for role in ("planner", "researcher", "solver", "single_policy"):
            graph.add_node(role, self._role_node(role))
        graph.add_node("search_or_read", self._execute_tool)
        graph.add_node("verify_finish", self._verify)
        graph.add_edge(START, "observe")
        graph.add_conditional_edges(
            "observe",
            self._choose_role,
            {
                "planner": "planner",
                "researcher": "researcher",
                "solver": "solver",
                "single_policy": "single_policy",
                "done": END,
            },
        )
        for role in ("planner", "researcher", "solver", "single_policy"):
            graph.add_conditional_edges(
                role,
                self._route_action,
                {"tool": "search_or_read", "finish": "verify_finish", "done": END},
            )
        graph.add_edge("search_or_read", "observe")
        graph.add_edge("verify_finish", "observe")
        self.graph = graph.compile(checkpointer=checkpointer)

    def _observe(self, state: ResearchState) -> dict[str, Any]:
        if state.get("termination"):
            return {}
        if state["steps_used"] >= self.config.max_steps:
            return {"termination": "step_budget_exhausted"}
        if self.evidence_state_mode == "off":
            return {"evidence_summary": None}
        # Only public question text and actually acquired tool observations.
        summary = derive_evidence_state(
            state["question"], state["observations"], version=self.evidence_state_mode
        ).public_summary()
        return {"evidence_summary": summary}

    def _choose_role(self, state: ResearchState) -> str:
        if state.get("termination"):
            return "done"
        if self.role_mode == "single":
            return "single_policy"
        if not state["steps_used"]:
            return "planner"
        if len(state["read_ids"]) < self.min_citations:
            return "researcher"
        return "solver"

    def _role_node(self, role: str):
        def decide(state: ResearchState) -> dict[str, Any]:
            role_policy = self.role_policies.get(role, self.policy)
            before = int(getattr(role_policy, "model_calls", 0))
            try:
                if role != "single_policy" and callable(
                    getattr(role_policy, "decide_with_system", None)
                ):
                    guidance = ROLE_GUIDANCE[role]
                    if state.get("evidence_summary"):
                        guidance += (
                            "\nNon-oracle lexical evidence proxy (NOT entailment): "
                            + json.dumps(state["evidence_summary"], sort_keys=True)
                        )
                    action = role_policy.decide_with_system(
                        state["question"],
                        list(state["observations"]),
                        self.config.max_steps - state["steps_used"],
                        SYSTEM + "\n" + guidance,
                    )
                else:
                    action = role_policy.decide(
                        state["question"],
                        list(state["observations"]),
                        self.config.max_steps - state["steps_used"],
                    )
                if not isinstance(action, Action):
                    raise TypeError("policy must return sidra.schema.Action")
                wire = action.to_dict()
                wire["citations"] = list(action.citations)
                step = {
                    "step": state["steps_used"],
                    "role": role,
                    "role_model_calls": (
                        int(getattr(role_policy, "model_calls", 0)) - before
                    ),
                    "action": wire,
                }
                if state.get("evidence_summary"):
                    step["evidence_state_before"] = state["evidence_summary"]
                return {
                    "pending_action": wire,
                    "steps": [*state["steps"], step],
                    "steps_used": state["steps_used"] + 1,
                }
            except Exception as exc:
                message = f"{type(exc).__name__}: {str(exc)[:200]}"
                return {
                    "termination": "error",
                    "error": message,
                    "pending_action": None,
                    "steps": [*state["steps"], {
                        "step": state["steps_used"], "role": role, "error": message
                    }],
                    "steps_used": state["steps_used"] + 1,
                }
        return decide

    @staticmethod
    def _route_action(state: ResearchState) -> str:
        if state.get("termination") or not state.get("pending_action"):
            return "done"
        return "finish" if state["pending_action"]["kind"] == "finish" else "tool"

    @staticmethod
    def _append_failure(state: ResearchState, reason: str) -> dict[str, Any]:
        return {
            "termination": "error",
            "error": reason[:220],
            "pending_action": None,
            "steps": [*state["steps"][:-1], {
                **state["steps"][-1], "error": reason[:220]
            }],
        }

    def _execute_tool(self, state: ResearchState) -> dict[str, Any]:
        if state["tool_calls"] >= self.config.max_tool_calls:
            return {
                "termination": "tool_budget_exhausted",
                "pending_action": None,
                "steps": [*state["steps"][:-1], {
                    **state["steps"][-1], "error": "tool budget exhausted"
                }],
            }
        action = Action.from_dict(state["pending_action"] or {})
        try:
            if action.kind == "search":
                if not action.query.strip():
                    raise ValueError("Search query must be non-empty")
                hits = self.index.search(
                    action.query, k=self.config.search_top_k, scope=state["scope"]
                )
                observation = {
                    "kind": "search", "query": action.query,
                    "hits": [hit.as_dict() for hit in hits],
                }
                return {
                    "observations": [*state["observations"], observation],
                    "seen_hit_ids": sorted(set(state["seen_hit_ids"]) | {
                        hit.doc_id for hit in hits
                    }),
                    "steps": [*state["steps"][:-1], {
                        **state["steps"][-1], "observation": observation
                    }],
                    "tool_calls": state["tool_calls"] + 1,
                    "search_calls": state["search_calls"] + 1,
                    "pending_action": None,
                }
            if action.kind == "read":
                if action.doc_id not in state["seen_hit_ids"]:
                    raise ValueError("read requires a doc_id returned by an earlier search")
                doc = self.docs[action.doc_id]
                if doc.scope not in {state["scope"], "global"}:
                    raise ValueError("read attempted outside question scope")
                observation = {
                    "kind": "read", "doc_id": doc.doc_id, "title": doc.title,
                    "text": doc.text[:self.config.max_read_chars],
                    "truncated": len(doc.text) > self.config.max_read_chars,
                }
                return {
                    "observations": [*state["observations"], observation],
                    "read_ids": sorted(set(state["read_ids"]) | {doc.doc_id}),
                    "steps": [*state["steps"][:-1], {
                        **state["steps"][-1], "observation": observation
                    }],
                    "tool_calls": state["tool_calls"] + 1,
                    "read_calls": state["read_calls"] + 1,
                    "pending_action": None,
                }
            raise ValueError("Tool node only accepts search or read")
        except Exception as exc:
            return self._append_failure(
                state, f"{type(exc).__name__}: {str(exc)[:200]}"
            )

    def _verify(self, state: ResearchState) -> dict[str, Any]:
        action = Action.from_dict(state["pending_action"] or {})
        if not action.answer.strip():
            return {
                "termination": "abstained", "pending_action": None,
                "steps": [*state["steps"][:-1], {
                    **state["steps"][-1],
                    "verification": {"approved": False, "reason": "empty_answer"}
                }],
            }
        unread = sorted(set(action.citations) - set(state["read_ids"]))
        if unread:
            reason = "unread_citations"
            detail = f"Unverified citations: {unread}; must read before cite"
        elif self.role_mode == "multi":
            audit = EvidenceAudit(self.min_citations).verify(
                action, state["observations"]
            )
            reason = audit.reason if not audit.approved else ""
            detail = reason
        else:
            reason, detail = "", ""
        if reason:
            entry = {
                **state["steps"][-1],
                "verification": {"approved": False, "reason": reason},
            }
            if (reason == "unread_citations" and not
                    self.config.recover_unread_citations):
                return {
                    "termination": "error",
                    "error": detail,
                    "steps": [*state["steps"][:-1], entry],
                    "pending_action": None,
                }
            if state["steps_used"] >= self.config.max_steps:
                return {
                    "termination": "error",
                    "error": detail,
                    "steps": [*state["steps"][:-1], entry],
                    "pending_action": None,
                }
            # Non-oracle feedback, not a fabricated answer or extra free step.
            feedback = {
                "kind": "validation_error", "code": reason,
                "read_doc_ids": list(state["read_ids"]),
                "unread_citations": unread,
                "instruction": (
                    "You may cite only doc IDs actually READ. "
                    "Search/READ additional documents or finish using only "
                    "READ citations. Do not invent citations."
                ),
            }
            return {
                "observations": [*state["observations"], feedback],
                "steps": [*state["steps"][:-1], {
                    **entry, "observation": feedback, "validation_failed": True
                }],
                "pending_action": None,
            }
        return {
            "termination": "finish",
            "answer": action.answer.strip(),
            "citations": list(dict.fromkeys(action.citations)),
            "pending_action": None,
            "steps": [*state["steps"][:-1], {
                **state["steps"][-1],
                "verification": {
                    "approved": True, "reason": (
                        "lexical_only" if self.role_mode == "multi"
                        else "read_before_cite_only"
                    )
                },
            }],
        }

    def run(self, question: Question) -> AgentTrace:
        if not question.question.strip():
            raise ValueError("question must not be blank")
        # DO NOT pass the Question object: it holds evaluator-only gold fields.
        initial: ResearchState = {
            "question_id": question.question_id,
            "question": question.question,
            "scope": question.scope,
            "observations": [],
            "steps": [],
            "seen_hit_ids": [],
            "read_ids": [],
            "evidence_summary": None,
            "steps_used": 0,
            "tool_calls": 0,
            "search_calls": 0,
            "read_calls": 0,
            "pending_action": None,
            "answer": "",
            "citations": [],
            "termination": "",
            "error": "",
        }
        thread_id = f"{self.thread_namespace}:{question.question_id}:{uuid4().hex}"
        self.last_thread_id = thread_id
        start = time.perf_counter()
        # Account for distinct model providers without double counting shared roles.
        policies = {id(p): p for p in (self.policy, *self.role_policies.values())}
        before = {key: {
            name: int(getattr(p, name, 0))
            for name in ("model_calls", "prompt_tokens", "completion_tokens")
        } for key, p in policies.items()}

        def usage_delta(name: str) -> int:
            return sum(int(getattr(p, name, 0)) - before[key][name]
                       for key, p in policies.items())
        final = self.graph.invoke(
            initial,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 6 * self.config.max_steps + 12,
            },
        )
        return AgentTrace(
            question_id=question.question_id,
            question=question.question,
            policy=(f"langgraph_{self.role_mode}_{self.policy.name}"
                    + ("_role_split" if self.role_policies else "")),
            steps=final["steps"],
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            termination=final.get("termination") or "step_budget_exhausted",
            tool_calls=final["tool_calls"],
            search_calls=final["search_calls"],
            read_calls=final["read_calls"],
            elapsed_ms=round((time.perf_counter() - start) * 1000),
            model_calls=usage_delta("model_calls"),
            prompt_tokens=usage_delta("prompt_tokens"),
            completion_tokens=usage_delta("completion_tokens"),
            error=final.get("error", ""),
        )
