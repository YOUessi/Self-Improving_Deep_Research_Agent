"""Non-oracle invalid-citation feedback keeps tool limits and labels sealed."""
from sidra.agent import AgentConfig, SearchAgent
from sidra.schema import Action, Document, Question


DOCS = [
    Document("a", "Alpha Bridges", "Alpha facts in Port city."),
    Document("b", "Beta Bridges", "Beta facts in Port city."),
]
Q = Question("sample-train", "What links Alpha and Beta?", "Port",
             supporting_doc_ids=("a", "b"), split="train")


class ReplayActions:
    name = "fixed_nonoracle_policy"

    def __init__(self, actions):
        self.todo = iter(actions)
        self.seen = []

    def decide(self, question, observations, remaining):
        self.seen.append((question, list(observations), remaining))
        return next(self.todo)


def test_invalid_finish_can_recover_only_after_reading_docs():
    policy = ReplayActions([
        Action("search", query="Bridges"),
        Action("read", doc_id="a"),
        Action("finish", answer="Port", citations=("a", "b")),
        Action("read", doc_id="b"),
        Action("finish", answer="Port", citations=("a", "b")),
    ])
    agent = SearchAgent(DOCS, policy,
                        AgentConfig(max_steps=5, max_tool_calls=3,
                                    recover_unread_citations=True))
    tr = agent.run(Q)
    assert tr.termination == "finish"
    assert tr.tool_calls == 3
    assert tr.read_calls == 2
    assert tr.answer == "Port"
    assert tr.citations == ["a", "b"]
    assert len(tr.steps) == 5
    bad = tr.steps[2]
    assert bad["validation_failed"] is True
    assert bad["observation"]["kind"] == "validation_error"
    assert bad["observation"]["code"] == "unread_citations"
    assert bad["observation"]["unread_citations"] == ["b"]
    assert bad["observation"]["read_doc_ids"] == ["a"]
    assert policy.seen[3][1][-1] == bad["observation"]
    for item in policy.seen:
        assert item[0] == Q.question
        assert "gold_answer" not in str(item)
        assert "supporting_doc_ids" not in str(item)


def test_strict_legacy_mode_still_terminates_on_invalid_finish():
    policy = ReplayActions([Action("finish", answer="Port", citations=("a",))])
    tr = SearchAgent(DOCS, policy).run(Q)
    assert tr.termination == "error"
    assert "Unverified citations" in tr.error
    assert len(tr.steps) == 1


def test_recovery_exhausts_steps_without_free_credit():
    policy = ReplayActions([Action("finish", answer="Port", citations=("b",))])
    tr = SearchAgent(DOCS, policy,
                     AgentConfig(max_steps=1, max_tool_calls=3,
                                 recover_unread_citations=True)).run(Q)
    assert tr.termination == "error"
    assert not tr.answer and not tr.citations
    assert len(tr.steps) == 1
    assert tr.steps[0]["validation_failed"]
    assert tr.tool_calls == 0


def test_recovery_cannot_evade_tool_budget():
    policy = ReplayActions([
        Action("search", query="Bridges"),
        Action("read", doc_id="a"),
        Action("finish", answer="Port", citations=("a", "b")),
        Action("read", doc_id="b"),
    ])
    tr = SearchAgent(DOCS, policy,
                     AgentConfig(max_steps=6, max_tool_calls=2,
                                 recover_unread_citations=True)).run(Q)
    assert tr.termination == "tool_budget_exhausted"
    assert tr.tool_calls == 2
    assert tr.read_calls == 1
    assert not tr.answer and not tr.citations
