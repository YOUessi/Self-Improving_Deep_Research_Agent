"""Three role-specialized policies sharing one bounded tool runtime.

A single underlying model instance may supply several *logical* agents. Role
instructions do not change access rights or expose evaluator gold answers.
"""

from __future__ import annotations

from sidra.evidence_audit import EvidenceAudit
from sidra.policies import SYSTEM
from sidra.schema import Action

ROLE_GUIDANCE = {
    "planner": "Planner role: start with a specific SEARCH action for the first needed fact.",
    "researcher": (
        "Researcher role: collect both hops. Choose READ for unread search hits "
        "or SEARCH for the next linked entity. Do not finish early."
    ),
    "solver": (
        "Solver role: finish only with a short supported answer and all relevant "
        "READ citations. Otherwise choose SEARCH or READ."
    ),
}


class MultiAgentPolicy:
    """Planner / Researcher / Solver followed by a source-only Verifier.

    The Verifier is not an answer oracle. Its lexical check cannot establish
    semantic entailment. Evaluate against labels outside this policy.
    """

    def __init__(self, model_policy, min_citations: int = 2):
        self.model_policy = model_policy
        self.audit = EvidenceAudit(min_citations)
        self.min_citations = min_citations
        self.name = "multi_role_" + model_policy.name
        self.last_role = ""
        self.last_verification = None
        self.last_model_calls = 0

    def _call(self, role: str, question: str, observations: list[dict], remaining: int) -> Action:
        self.last_model_calls += 1
        role_decide = getattr(self.model_policy, "decide_with_system", None)
        if callable(role_decide):
            return role_decide(question, observations, remaining, SYSTEM + "\n" + ROLE_GUIDANCE[role])
        return self.model_policy.decide(question, observations, remaining)

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        self.last_verification = None
        self.last_model_calls = 0
        read_ids = {o["doc_id"] for o in observations if o.get("kind") == "read"}
        if not observations:
            role = "planner"
        elif len(read_ids) < self.min_citations and remaining > 1:
            role = "researcher"
        else:
            role = "solver"
        self.last_role = role
        candidate = self._call(role, question, observations, remaining)

        if role == "planner" and candidate.kind != "search":
            self.last_verification = {"approved": False, "reason": "planner_must_search"}
            return Action(kind="finish", answer="")
        if role == "researcher" and candidate.kind == "finish":
            self.last_verification = {"approved": False, "reason": "researcher_cannot_finish"}
            return Action(kind="finish", answer="")
        if candidate.kind != "finish":
            return candidate

        decision = self.audit.verify(candidate, observations)
        self.last_verification = {"approved": decision.approved, "reason": decision.reason}
        if decision.approved:
            return candidate
        if remaining > 1:
            self.last_role = "researcher"
            retry = self._call("researcher", question, observations, remaining)
            if retry.kind in {"read", "search"}:
                return retry
        return Action(kind="finish", answer="")
