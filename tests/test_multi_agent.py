"""Multi-role coordination and oracle-free evidence audit tests."""
from sidra.agent import SearchAgent
from sidra.evidence_audit import EvidenceAudit
from sidra.multi_agent import MultiAgentPolicy
from sidra.policies import FixtureMultiHopPolicy
from sidra.schema import Action, Document, Question


def test_multi_role_fixture_pipeline(fixture_data):
    docs, qs = fixture_data
    q = next(x for x in qs if x.question_id == "fixture_014")
    agent = SearchAgent(docs, MultiAgentPolicy(FixtureMultiHopPolicy()))
    trace = agent.run(q)
    assert trace.answer == q.gold_answer
    assert trace.termination == "finish"
    assert set(trace.citations) == set(q.supporting_doc_ids)
    assert trace.tool_calls == 4
    assert [step["role"] for step in trace.steps] == [
        "planner", "researcher", "researcher", "researcher", "solver"
    ]
    assert trace.steps[-1]["verification"]["approved"] is True


def test_planner_cannot_skip_search():
    class Premature:
        name = "premature"
        def decide(self, question, observations, remaining):
            return Action(kind="finish", answer="MadeUp")
    trace = SearchAgent([Document("a", "A", "Fact")],
                        MultiAgentPolicy(Premature())).run(
        Question("q", "What?", "MadeUp"))
    assert trace.termination == "abstained"
    assert trace.steps[-1]["verification"]["reason"] == "planner_must_search"


def test_two_distinct_sources_required():
    audit = EvidenceAudit(min_citations=2)
    one = [dict(kind="read", doc_id="a", text="Ore014")]
    assert audit.verify(Action("finish", answer="Ore014", citations=("a",)), one).approved is False
    assert audit.verify(Action("finish", answer="Ore014", citations=("a", "b")), one).reason == "unread_citations"
    both = one + [dict(kind="read", doc_id="b", text="Extra context")]
    assert audit.verify(Action("finish", answer="Ore014", citations=("a", "b")), both).approved is True
    assert audit.verify(Action("finish", answer="Unsupported", citations=("a", "b")), both).approved is False


def test_role_model_sees_no_gold(fixture_data):
    docs, qs = fixture_data
    q = next(x for x in qs if x.question_id == "fixture_014")

    class Spy:
        name = "scripted_roles"
        def __init__(self):
            self.seen = []
        def decide_with_system(self, question, observations, remaining, system):
            self.seen.append((question, system, observations))
            k = len(self.seen)
            sequence = [
                Action("search", query="Explorer Mara014"),
                Action("read", doc_id="explorer_014"),
                Action("search", query="Vale014"),
                Action("read", doc_id="region_014"),
                Action("finish", answer="Ore014",
                       citations=("explorer_014", "region_014")),
            ]
            return sequence[k - 1]

    spy = Spy()
    trace = SearchAgent(docs, MultiAgentPolicy(spy)).run(q)
    assert trace.termination == "finish"
    assert len(spy.seen) == 5
    assert "Planner role:" in spy.seen[0][1]
    assert "Researcher role:" in spy.seen[1][1]
    assert "Solver role:" in spy.seen[-1][1]
    for question, system, observations in spy.seen:
        assert question == q.question
        assert "gold_answer" not in system
        assert "gold_answer" not in str(observations)


def test_researcher_early_answer_blocked():
    class TooEarly:
        name = "too_early"
        def __init__(self):
            self.i = 0
        def decide(self, question, observations, remaining):
            self.i += 1
            return Action("search", query="fact") if self.i == 1 else Action(
                "finish", answer="fact", citations=())
    trace = SearchAgent([Document("x", "fact", "fact")],
                        MultiAgentPolicy(TooEarly())).run(Question("q", "fact?", "fact"))
    assert trace.termination == "abstained"
    assert trace.steps[-1]["verification"]["reason"] == "researcher_cannot_finish"
