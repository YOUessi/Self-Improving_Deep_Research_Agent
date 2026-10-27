"""Gold-blind, frontier-first second-hop planning (opt-in M7 prototype).

This does NOT use HotpotQA answer/support labels or free/hidden index lookups.
The frontier consists ONLY of search hits already returned by SearchAgent.
Prefer reading a visibly linked document to paying for a second query;
otherwise issue the ordinary billed SEARCH. Both variants keep the same
max_steps/max_tool_calls as their baseline. Not enabled by default.
"""
from __future__ import annotations

from dataclasses import dataclass

from sidra.bridge import BridgeCandidate, next_bridge_query, normalize
from sidra.bridge_policy import choose_unread_hit
from sidra.retrieval import tokenize
from sidra.schema import Action


@dataclass(frozen=True)
class NextHopPlan:
    action: Action
    reason: str
    candidate: BridgeCandidate | None


def _phrase_in(phrase: str, text: str) -> bool:
    """Whole-token contiguous phrase match, not substring e.g. 'art' in 'earth'."""
    needle = tokenize(phrase)
    if len(needle) < 2 or not any(len(t) >= 4 for t in needle):
        return False
    hay = tokenize(text)
    width = len(needle)
    return any(hay[i:i + width] == needle for i in range(len(hay) - width + 1))


def plan_next_hop(
    question: str,
    observations: list[dict],
    *,
    remaining_steps: int,
    remaining_tool_calls: int,
) -> NextHopPlan | None:
    """Plan a single safe next tool action from one actual READ observation.

    remaining_steps includes the final FINISH; SEARCH requires another READ
    and FINISH, so it needs >=3 steps and >=2 remaining tool calls.
    """
    reads = [o for o in observations if o.get("kind") == "read" and o.get("doc_id")]
    searches = [o for o in observations if o.get("kind") == "search"]
    if len(reads) != 1 or not searches or remaining_steps < 2 or remaining_tool_calls < 1:
        return None
    last_read_index = max(i for i, o in enumerate(observations) if o.get("kind") == "read")
    if any(o.get("kind") == "search" for o in observations[last_read_index + 1:]):
        return None

    proposal = next_bridge_query(question, observations)
    read = reads[0]
    already_read = {str(o["doc_id"]) for o in reads}
    # A frontier document may appear in multiple SEARCH lists. Never invent an ID.
    exposed: dict[str, tuple[int, dict]] = {}
    for obs in searches:
        for rank, hit in enumerate(obs.get("hits") or []):
            doc_id = str(hit.get("doc_id", ""))
            if doc_id and doc_id not in already_read and doc_id not in exposed:
                exposed[doc_id] = (rank, hit)

    candidates: list[tuple[float, int, str, str]] = []
    for doc_id, (rank, hit) in exposed.items():
        title = str(hit.get("title", "")).strip()
        if not title:
            continue
        if proposal is not None and normalize(title) == normalize(proposal.query):
            confidence, reason = 4.0, "frontier_exact_bridge_entity"
        elif _phrase_in(title, str(read.get("text", ""))):
            confidence, reason = 3.0, "frontier_title_mentioned_in_read"
        elif _phrase_in(title, question):
            confidence, reason = 2.0, "frontier_title_mentioned_in_question"
        else:
            continue
        candidates.append((-confidence, rank, doc_id, reason))
    if candidates:
        _, _, doc_id, reason = min(candidates)
        return NextHopPlan(Action("read", doc_id=doc_id), reason, proposal)

    if proposal and remaining_steps >= 3 and remaining_tool_calls >= 2:
        previous = {normalize(str(o.get("query", ""))) for o in searches}
        if normalize(proposal.query) not in previous:
            return NextHopPlan(Action("search", query=proposal.query),
                               "observed_second_hop_search", proposal)
    return None


class FrontierFirstBridgePolicy:
    """Opt-in wrapper: replacement for BridgeGroundedPolicy in paired tests.

    The constructor budget MUST equal the SearchAgent's max_tool_calls.
    A policy can still choose other tool calls via the underlying base policy.
    """
    def __init__(self, grounded_policy, *, max_tool_calls: int):
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.base = grounded_policy
        self.max_tool_calls = max_tool_calls
        self.name = "frontier_bridge_" + grounded_policy.name
        self.model_name = getattr(grounded_policy, "model_name", None)
        self.adapter_path = getattr(grounded_policy, "adapter_path", None)
        self.last_role = ""
        self.last_verification = None
        self.last_model_calls = 0

    @property
    def model_calls(self) -> int:
        return int(getattr(self.base, "model_calls", 0))

    @property
    def prompt_tokens(self) -> int:
        return int(getattr(self.base, "prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(getattr(self.base, "completion_tokens", 0))

    def _override(self, action: Action, reason: str) -> Action:
        self.last_role = "frontier_next_hop"
        self.last_verification = {"approved": False, "reason": reason}
        self.last_model_calls = 0
        return action

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        used = sum(o.get("kind") in {"search", "read"} for o in observations)
        tool_remaining = self.max_tool_calls - used
        reads = [o for o in observations if o.get("kind") == "read"]
        searches = [o for o in observations if o.get("kind") == "search"]

        if len(reads) == 1:
            last_read = max(i for i, o in enumerate(observations) if o.get("kind") == "read")
            has_new_search = any(o.get("kind") == "search"
                                 for o in observations[last_read + 1:])
            if has_new_search and remaining >= 2 and tool_remaining >= 1:
                choice = choose_unread_hit(question, observations)
                if choice:
                    return self._override(Action("read", doc_id=choice),
                                          "observed_second_hop_frontier_read")
            else:
                plan = plan_next_hop(question, observations,
                                     remaining_steps=remaining,
                                     remaining_tool_calls=tool_remaining)
                if plan:
                    return self._override(plan.action, plan.reason)

        action = self.base.decide(question, observations, remaining)
        self.last_role = getattr(self.base, "last_role", "grounded")
        self.last_verification = getattr(self.base, "last_verification", None)
        self.last_model_calls = getattr(self.base, "last_model_calls", 0)
        return action
