"""Regression tests for actual Qwen3 M3 multi-role failure categories.

No gold labels are sent to any policy. Retries are bounded and counted.
"""
from sidra.multi_agent import MultiAgentPolicy
from sidra.schema import Action


def test_invalid_json_is_retried_once_with_extra_model_call():
    class FailingFirst:
        name = "failing_json"
        def __init__(self):
            self.calls = 0
            self.instructions = []
        def decide_with_system(self, question, observations, remaining, system):
            self.calls += 1
            self.instructions.append(system)
            if self.calls == 1:
                raise ValueError("Model output did not contain a valid agent action JSON object")
            return Action("search", query="Alpha")
    base = FailingFirst()
    agent = MultiAgentPolicy(base)
    action = agent.decide("a question", [], remaining=5)
    assert action.kind == "search"
    assert agent.model_calls == 2
    assert agent.last_model_calls == 2
    assert "ONE VALID JSON" in base.instructions[-1]


def test_duplicate_read_replans_without_repeating_tool_action():
    class RepeatThenSearch:
        name = "duplicate_read"
        def __init__(self):
            self.calls = 0
        def decide_with_system(self, question, observations, remaining, system):
            self.calls += 1
            return Action("read", doc_id="doc-a") if self.calls == 1 else Action(
                "search", query="next entity")
    policy = MultiAgentPolicy(RepeatThenSearch())
    observations = [{"kind":"read","doc_id":"doc-a","text":"some fact"}]
    answer = policy.decide("question", observations, remaining=4)
    assert answer.kind == "search"
    assert policy.model_calls == 2
    assert policy.last_verification["reason"] == "duplicate_read_replanned"


def test_premature_finish_replans_without_gold_answers():
    class PrematureThenSearch:
        name = "premature"
        def __init__(self): self.calls = 0
        def decide_with_system(self, question, observations, remaining, system):
            self.calls += 1
            if self.calls == 1:
                return Action("finish", answer="guess", citations=("doc-a",))
            return Action("search", query="find independent confirmation")
    p = MultiAgentPolicy(PrematureThenSearch())
    action = p.decide("question", [{"kind":"read","doc_id":"doc-a","text":"fact"}], remaining=4)
    assert action.kind == "search"
    assert p.model_calls == 2
    assert p.last_verification["reason"] == "premature_finish_replanned"
