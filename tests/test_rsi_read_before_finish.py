"""E-030 CPU integration fixtures: NOT real Qwen3 inference or RSI evidence."""
import pytest

from sidra.agent import AgentConfig, SearchAgent
from sidra.schema import Action, Document, Question
from sidra.rsi_read_before_finish import ObservedReadBeforeFinishPolicy


class BasicFakePolicy:
    name = "synthetic_test_fixture_only"

    def __init__(self):
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def decide(self, question, observations, remaining):
        assert "EVALUATOR_HIDDEN_LABEL" not in question
        assert not any("EVALUATOR_HIDDEN_LABEL" in str(x) for x in observations)
        self.model_calls += 1
        self.prompt_tokens += 10
        self.completion_tokens += 5
        if not observations:
            return Action(kind="search",
                          query="Old Mill Group Blue Garden Foundation Mountain")
        read_ids = [o["doc_id"] for o in observations if o["kind"] == "read"]
        if not read_ids:
            search = next(o for o in observations if o["kind"] == "search")
            old = next(x for x in search["hits"] if x["title"] == "Old Mill Group")
            return Action(kind="read", doc_id=old["doc_id"])
        return Action(kind="finish", answer="uncertain", citations=tuple(read_ids))


def make_docs(many=True):
    docs = [
        Document("d-old", "Old Mill Group",
                 "Old Mill Group partners with Blue Garden Foundation", "s"),
        Document("d-bridge", "Blue Garden Foundation",
                 "Blue Garden Foundation manages a local community garden", "s"),
    ]
    if many:
        docs.append(Document(
            "d-irrelevant", "Mountain Cartography",
            "Mountain Cartography publishes mountain terrain maps", "s"))
    return docs


def question():
    return Question(
        "test-train-qid", "Who partners with Old Mill Group?",
        "EVALUATOR_HIDDEN_LABEL", ("d-bridge",), split="train", scope="s")


def test_guard_inserts_one_real_searchagent_read_with_same_global_budget():
    docs = make_docs()
    cfg = AgentConfig(max_steps=6, max_tool_calls=5, search_top_k=3)
    control = SearchAgent(docs, BasicFakePolicy(), cfg).run(question())
    base = BasicFakePolicy()
    wrapped = ObservedReadBeforeFinishPolicy(
        base, max_steps=cfg.max_steps, max_tool_calls=cfg.max_tool_calls)
    treated = SearchAgent(docs, wrapped, cfg).run(question())
    assert control.termination == treated.termination == "finish"
    assert control.search_calls == treated.search_calls == 1
    assert control.read_calls == 1 and treated.read_calls == 2
    assert control.tool_calls == 2 and treated.tool_calls == 3
    assert treated.tool_calls <= cfg.max_tool_calls
    assert len(treated.steps) <= cfg.max_steps
    assert [s["action"]["kind"] for s in treated.steps] == [
        "search", "read", "read", "finish"]
    assert [s["action"]["kind"] for s in control.steps] == [
        "search", "read", "finish"]
    assert treated.steps[2]["role"] == "observed_read_before_finish"
    assert treated.steps[2]["role_model_calls"] == 0
    assert treated.steps[2]["verification"]["gold_used"] is False
    assert treated.steps[2]["action"]["doc_id"] == "d-bridge"
    assert treated.model_calls == base.model_calls == 3
    assert treated.answer == "uncertain"
    assert "EVALUATOR_HIDDEN_LABEL" not in str(treated.steps)


def test_guard_does_nothing_if_only_one_alternative_document():
    cfg = AgentConfig(max_steps=6, max_tool_calls=5, search_top_k=3)
    base = BasicFakePolicy()
    guard = ObservedReadBeforeFinishPolicy(base)
    trace = SearchAgent(make_docs(many=False), guard, cfg).run(question())
    assert trace.termination == "finish"
    assert trace.read_calls == 1
    assert trace.steps[-1]["action"]["kind"] == "finish"


def test_guard_does_nothing_without_step_budget_for_read_and_finish():
    cfg = AgentConfig(max_steps=3, max_tool_calls=5, search_top_k=3)
    base = BasicFakePolicy()
    guard = ObservedReadBeforeFinishPolicy(
        base, max_steps=cfg.max_steps, max_tool_calls=cfg.max_tool_calls)
    trace = SearchAgent(make_docs(), guard, cfg).run(question())
    assert trace.read_calls == 1
    assert trace.termination == "finish"
    assert len(trace.steps) == 3


def test_guard_does_not_add_read_when_no_tool_budget():
    cfg = AgentConfig(max_steps=6, max_tool_calls=2, search_top_k=3)
    guard = ObservedReadBeforeFinishPolicy(
        BasicFakePolicy(), max_steps=6, max_tool_calls=2)
    trace = SearchAgent(make_docs(), guard, cfg).run(question())
    assert trace.tool_calls == 2
    assert trace.read_calls == 1
    assert trace.termination == "finish"


def test_guard_rejects_bad_constructor_budget():
    with pytest.raises(ValueError):
        ObservedReadBeforeFinishPolicy(BasicFakePolicy(), max_steps=1)
    with pytest.raises(ValueError):
        ObservedReadBeforeFinishPolicy(BasicFakePolicy(), max_tool_calls=0)


def test_gold_field_cannot_enter_policy_as_search_observation():
    base = BasicFakePolicy()
    guard = ObservedReadBeforeFinishPolicy(base)
    from test_rsi_observed_read import observed
    tainted = observed()
    tainted[0]["gold_answer"] = "EVALUATOR_HIDDEN_LABEL"
    with pytest.raises(ValueError, match="forbidden search"):
        guard.decide("Who partners with Old Mill Group?", tainted, 4)
    assert base.model_calls == 0


def test_unchanged_base_when_guard_not_eligible():
    base = BasicFakePolicy()
    guard = ObservedReadBeforeFinishPolicy(base)
    action = guard.decide("Who partners with Old Mill Group?", [], 6)
    assert action.kind == "search"
    assert base.model_calls == 1
    assert guard.model_calls == 1
