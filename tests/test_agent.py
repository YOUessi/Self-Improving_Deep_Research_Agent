from sidra.agent import AgentConfig, SearchAgent
from sidra.policies import FixtureMultiHopPolicy, FixtureOneHopPolicy, _json_object
from sidra.schema import Action, Document, Question


class FakePolicy:
    name = "fake"

    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, question, observations, remaining):
        return next(self.actions)


def test_fixture_two_hop_with_earned_citations(fixture_data):
    docs, questions = fixture_data
    agent = SearchAgent(docs, FixtureMultiHopPolicy())
    q = next(q for q in questions if q.question_id == "fixture_014")
    trace = agent.run(q)
    assert trace.answer == q.gold_answer
    assert set(trace.citations) == set(q.supporting_doc_ids)
    assert trace.tool_calls == 4
    assert trace.termination == "finish"
    assert len(trace.steps) == 5
    # No oracle labels may appear as fields in the model-facing action history.
    assert all("gold_answer" not in step for step in trace.steps)


def test_one_hop_cannot_manufacture_evidence(fixture_data):
    docs, questions = fixture_data
    trace = SearchAgent(docs, FixtureOneHopPolicy()).run(questions[0])
    assert not trace.answer
    assert not trace.citations
    assert trace.termination == "abstained"


def test_forbid_unread_citation():
    docs = [Document("a", "Alpha", "Alpha mineral is Gold")]
    q = Question("q", "Question", "Gold")
    policy = FakePolicy([Action("finish", answer="Gold", citations=("a",))])
    trace = SearchAgent(docs, policy).run(q)
    assert trace.termination == "error"
    assert trace.answer == ""
    assert "Unverified citations" in trace.error


def test_read_requires_search_hit():
    docs = [Document("a", "Alpha", "Alpha mineral is Gold")]
    policy = FakePolicy([Action("read", doc_id="a")])
    trace = SearchAgent(docs, policy).run(Question("q", "Question", "Gold"))
    assert trace.termination == "error"
    assert "earlier search" in trace.error


def test_exhaust_tool_budget():
    docs = [Document("a", "Alpha", "Alpha mineral is Gold")]
    policy = FakePolicy([Action("search", query="Alpha")] * 5)
    trace = SearchAgent(docs, policy, AgentConfig(max_steps=5, max_tool_calls=2)).run(
        Question("q", "Question", "Gold"))
    assert trace.tool_calls == 2
    assert trace.termination == "tool_budget_exhausted"


def test_malformed_action_json_is_rejected():
    assert _json_object('```json\n{"kind":"search","query":"Alpha"}\n```')["query"] == "Alpha"
    assert _json_object('prefix {"kind":"finish","answer":"OK","citations":[]}')["kind"] == "finish"
