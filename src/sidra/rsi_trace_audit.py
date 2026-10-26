"""Fail-closed, source-replayed audit of SIDRA search/read/finish trajectories.

The audit verifies that *recorded* tool observations agree with the pinned
corpus + BM25 algorithm and SearchAgent budget/transition rules. It prevents
training on fabricated or tampered citations, but cannot prove a tool call was
actually emitted by model weights: for that, also bind the inference server's
signed/logged request and checkpoint provenance externally.

The audit sees no evaluator gold answers or supporting document IDs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from sidra.agent import AgentConfig
from sidra.retrieval import BM25Index
from sidra.schema import Action, AgentTrace, Document, Question


@dataclass(frozen=True)
class TraceAudit:
    structurally_valid: bool
    complete_legal_finish: bool
    reason: str
    checked_searches: int
    checked_reads: int
    checked_tool_calls: int
    checked_steps: int

    def as_dict(self) -> dict:
        return asdict(self)


def audit_tool_trace(
    question: Question, trace: AgentTrace, documents: list[Document],
    *, config: AgentConfig | None = None,
) -> TraceAudit:
    """Replay the tool evidence and validate transition/counter consistency.

    Unlike a permissive score over a supplied answer and fake citations,
    every READ observation MUST exactly match the corpus text the tool would
    have returned, and every SEARCH hit/score/snippet MUST match BM25. A valid
    finished trace must cite only actually read documents and correspond
    exactly to a committed FINISH action. Error/abstain traces may still be
    structurally valid but never become verified positive training samples.
    """
    config=config or AgentConfig()
    searched: set[str] = set()
    read: set[str] = set()
    search_count=0
    read_count=0
    tool_count=0
    steps=0

    def failure(reason: str) -> TraceAudit:
        return TraceAudit(
            structurally_valid=False,complete_legal_finish=False,
            reason=reason,checked_searches=search_count,
            checked_reads=read_count,checked_tool_calls=tool_count,
            checked_steps=steps,
        )

    if question.split != "train":
        return failure("train_only_verifier")
    if question.question_id != trace.question_id or question.question != trace.question:
        return failure("question_identity_mismatch")
    if len(trace.steps)>config.max_steps or not trace.steps:
        return failure("invalid_step_count")
    docmap={d.doc_id:d for d in documents}
    if len(docmap)!=len(documents) or not documents:
        return failure("invalid_corpus")
    # This index takes only public tool corpus text, never gold answers/IDs.
    index=BM25Index(documents,title_boost=config.title_boost)
    end_status=""
    for expected_step,entry in enumerate(trace.steps):
        if not isinstance(entry,dict) or entry.get("step")!=expected_step:
            return failure("step_index_or_shape_mismatch")
        steps+=1
        # SearchAgent records terminal runtime exceptions as an error-only
        # event without including the invalid action in the trace.
        if "action" not in entry:
            if (expected_step==len(trace.steps)-1
                and trace.termination=="error"
                and isinstance(entry.get("error"),str)
                and trace.error==entry["error"]):
                end_status="error"
                continue
            return failure("missing_action_on_nonerror_step")
        try:
            action=Action.from_dict(entry["action"])
        except (TypeError,ValueError,AttributeError,KeyError):
            return failure("malformed_action_record")
        if action.to_dict()!={
            **{"kind":action.kind,"query":action.query,"doc_id":action.doc_id,
               "answer":action.answer,"citations":action.citations},
        }:
            # This branch is intentionally harmless (tuple serialization can
            # vary); real legality follows from Action/from_dict and tool replay.
            pass
        if action.kind == "finish":
            if "observation" in entry:
                return failure("finish_has_tool_observation")
            if set(action.citations)-read:
                # In SearchAgent this case is a terminal error-only event,
                # not a successfully committed finish.
                return failure("finish_cites_unread_source")
            if trace.answer != action.answer.strip():
                return failure("finish_answer_mismatch")
            if trace.citations != list(dict.fromkeys(action.citations)):
                return failure("finish_citations_mismatch")
            if trace.termination != ("finish" if trace.answer else "abstained"):
                return failure("finish_termination_mismatch")
            if expected_step!=len(trace.steps)-1:
                return failure("actions_after_finish")
            end_status=trace.termination
            continue
        if tool_count>=config.max_tool_calls:
            # The real agent refuses the over-budget action without exposing
            # a tool observation; accept only a terminal budget marker.
            if ("observation" in entry
                or trace.termination!="tool_budget_exhausted"
                or expected_step!=len(trace.steps)-1
                or entry.get("error")!="tool budget exhausted"):
                return failure("tool_budget_exceeded")
            end_status="tool_budget_exhausted"
            continue
        if action.kind=="search":
            if not action.query.strip():
                return failure("empty_search")
            expected=index.search(
                action.query,k=config.search_top_k,scope=question.scope)
            observed=entry.get("observation")
            if not isinstance(observed,dict):
                return failure("search_without_observation")
            if set(observed)!={"kind","query","hits"} or (
                observed["kind"]!="search" or observed["query"]!=action.query):
                return failure("search_observation_metadata_changed")
            hits=observed["hits"]
            if not isinstance(hits,list) or len(hits)!=len(expected):
                return failure("search_hit_count_mismatch")
            for actual_hit,correct in zip(hits,expected):
                if not isinstance(actual_hit,dict):
                    return failure("malformed_search_hit")
                canonical=correct.as_dict()
                if set(actual_hit)!=set(canonical):
                    return failure("search_hit_fields_mismatch")
                for field in ("doc_id","title","snippet"):
                    if actual_hit[field]!=canonical[field]:
                        return failure("search_hit_forgery")
                try:
                    delta=abs(float(actual_hit["score"])-canonical["score"])
                except (TypeError,ValueError):
                    return failure("search_score_invalid")
                if not math.isfinite(delta) or delta>1e-5:
                    return failure("search_rank_or_score_forgery")
            searched.update(h.doc_id for h in expected)
            search_count+=1
        elif action.kind=="read":
            if action.doc_id not in searched:
                return failure("read_before_search")
            doc=docmap.get(action.doc_id)
            if doc is None or doc.scope not in (question.scope,"global"):
                return failure("read_out_of_scope")
            obs=entry.get("observation")
            expected_obs={
                "kind":"read","doc_id":doc.doc_id,"title":doc.title,
                "text":doc.text[:config.max_read_chars],
                "truncated":len(doc.text)>config.max_read_chars,
            }
            if obs != expected_obs:
                return failure("read_observation_forgery")
            read.add(doc.doc_id)
            read_count+=1
        else:
            return failure("unsupported_action")
        tool_count+=1
        if "error" in entry:
            return failure("tool_entry_contains_error")

    if (trace.search_calls!=search_count or trace.read_calls!=read_count
        or trace.tool_calls!=tool_count):
        return failure("tool_counter_mismatch")
    if not end_status:
        if (trace.termination=="step_budget_exhausted"
            and len(trace.steps)==config.max_steps):
            end_status="step_budget_exhausted"
        else:
            return failure("invalid_or_unaccounted_termination")
    if trace.termination!=end_status:
        return failure("termination_mismatch")
    if end_status not in ("finish","abstained"):
        if trace.answer or trace.citations:
            return failure("partial_trace_claims_final_answer")
    complete=(end_status=="finish" and bool(trace.answer)
              and bool(trace.citations) and not trace.error)
    return TraceAudit(
        structurally_valid=True,complete_legal_finish=complete,
        reason="tool_replay_verified" if complete else (
            "nonpositive_"+end_status),
        checked_searches=search_count,checked_reads=read_count,
        checked_tool_calls=tool_count,checked_steps=steps,
    )
