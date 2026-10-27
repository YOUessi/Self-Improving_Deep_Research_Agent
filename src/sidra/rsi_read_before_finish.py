"""E-030 opt-in, gold-blind SearchAgent guard against one-READ premature FINISH.

This is a rule-based experimental wrapper, not a trained policy and NEVER a
default deployment policy. READ costs normal SearchAgent budget. This module
must be usable by a model-only Solver process without importing evaluator
TRAIN data, a trusted corpus mapping, or reward/Verifier labels.
"""
from __future__ import annotations

from sidra.rsi_observed_read import choose_observed_second_read
from sidra.schema import Action


class ObservedReadBeforeFinishPolicy:
    """Insert at most one ordinary READ of a previously exposed document.

    Call with the same max_steps/max_tool_calls as the receiving SearchAgent.
    The wrapper cannot amend the model's real tool budget or invent new docs.
    """

    def __init__(
        self, base, *, max_steps: int = 6, max_tool_calls: int = 5
    ) -> None:
        if (type(max_steps) is not int or type(max_tool_calls) is not int
            or max_steps < 2 or max_tool_calls < 1):
            raise ValueError("invalid SearchAgent tool budget")
        self.base = base
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.name = "optin_read_before_finish_" + str(base.name)
        self.model_name = getattr(base, "model_name", None)
        self.adapter_path = getattr(base, "adapter_path", None)
        self.last_role = ""
        self.last_model_calls = 0
        self.last_verification = None

    @property
    def model_calls(self) -> int:
        return int(getattr(self.base, "model_calls", 0))

    @property
    def prompt_tokens(self) -> int:
        return int(getattr(self.base, "prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(getattr(self.base, "completion_tokens", 0))

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        # Only after exactly one actual READ; thereafter the guard is inert.
        reads = sum(o.get("kind") == "read" for o in observations)
        if reads == 1 and remaining == self.max_steps - len(observations):
            proposal = choose_observed_second_read(
                question, observations,
                max_steps=self.max_steps, max_tool_calls=self.max_tool_calls
            )
            if proposal:
                self.last_role = "observed_read_before_finish"
                self.last_model_calls = 0
                self.last_verification = {
                    "approved": True,
                    "reason": "observed_only_single_extra_read",
                    "legal_frontier_size": proposal.visible_candidates,
                    "gold_used": False,
                }
                return Action(kind="read", doc_id=proposal.doc_id)
        action = self.base.decide(question, observations, remaining)
        self.last_role = getattr(self.base, "last_role", "solver")
        self.last_model_calls = getattr(self.base, "last_model_calls", 1)
        self.last_verification = getattr(self.base, "last_verification", None)
        return action
