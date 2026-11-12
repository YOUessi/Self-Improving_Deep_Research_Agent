"""Real LangGraph node and edge integration tests: no model API or GPU."""
import json

import pytest

pytest.importorskip("langgraph")

from sidra.agent import AgentConfig
from sidra.experiment import evaluate
from sidra.io import save_jsonl
from sidra.langgraph_agent import GraphResearchAgent
from sidra.policies import FixtureMultiHopPolicy
from sidra.schema import Action, Document, Question
from sidra.train_reward import score_train_rollout


def corpus():
    docs = [
        Document("p", "Explorer Atlas7", "Explorer Atlas7 was born in Region8"),
        Document("r", "Region Region8", "Region Region8 primary mineral is Quartz9"),
    ]
    q = Question(
        "qa-01", "What primary mineral is in the region where explorer Atlas7 was born?",
        "Quartz9", ("p", "r"), split="train",
    )
    return docs, q


class ScriptedPolicy:
    name = "scripted_no_gold"

    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def decide(self, question, observations, remaining):
        self.calls.append((question, list(observations), remaining))
        return next(self.actions)


def test_two_hop_graph_roles_real_bm25_and_nonoracle_verifier():
    docs, q = corpus()
    runner = GraphResearchAgent(
        docs, FixtureMultiHopPolicy(), AgentConfig(max_steps=8, max_tool_calls=7),
        role_mode="multi", min_citations=2,
    )
    trace = runner.run(q)
    assert trace.termination == "finish", trace.error
    assert trace.answer == "Quartz9"
    assert set(trace.citations) == {"p", "r"}
    assert trace.tool_calls == 4
    assert trace.search_calls == trace.read_calls == 2
    assert [s["action"]["kind"] for s in trace.steps] == [
        "search", "read", "search", "read", "finish",
    ]
    assert {s["role"] for s in trace.steps} == {"planner", "researcher", "solver"}
    assert trace.steps[-1]["verification"] == {
        "approved": True, "reason": "lexical_only",
    }


def test_gold_never_passed_to_policy_or_checkpoint():
    from langgraph.checkpoint.memory import InMemorySaver

    docs, q = corpus()
    q = Question(q.question_id, q.question, "HIDDEN_GOLD_DO_NOT_SEND",
                 q.supporting_doc_ids, split="test")
    policy = ScriptedPolicy([Action("finish", answer="")])
    saver = InMemorySaver()
    runner = GraphResearchAgent(docs, policy, checkpointer=saver)
    trace = runner.run(q)
    assert trace.termination == "abstained"
    assert "HIDDEN_GOLD_DO_NOT_SEND" not in repr(policy.calls)
    persisted = runner.graph.get_state(
        {"configurable": {"thread_id": runner.last_thread_id}}
    ).values
    assert "gold_answer" not in persisted
    assert "supporting_doc_ids" not in persisted
    assert "HIDDEN_GOLD_DO_NOT_SEND" not in repr(persisted)


def test_read_must_follow_scoped_search():
    docs, q = corpus()
    runner = GraphResearchAgent(docs, ScriptedPolicy([Action("read", doc_id="r")]))
    trace = runner.run(q)
    assert trace.termination == "error"
    assert "earlier search" in trace.error
    assert trace.tool_calls == 0


def test_unread_citation_rejected_without_oracle():
    docs, q = corpus()
    runner = GraphResearchAgent(
        docs, ScriptedPolicy([Action("finish", answer="Quartz9",
                                    citations=("r",))]))
    trace = runner.run(q)
    assert trace.termination == "error"
    assert trace.answer == ""
    assert "Unverified citations" in trace.error


def test_verifier_feedback_can_replan_within_same_budget():
    docs, q = corpus()
    policy = ScriptedPolicy([
        Action("search", query="Explorer Atlas7"),
        Action("read", doc_id="p"),
        Action("search", query="Region8"),
        Action("read", doc_id="r"),
        Action("finish", answer="Quartz9", citations=("r",)),
        Action("finish", answer="Quartz9", citations=("p", "r")),
    ])
    runner = GraphResearchAgent(
        docs, policy, AgentConfig(max_steps=7, max_tool_calls=5),
        role_mode="multi", min_citations=2,
    )
    trace = runner.run(q)
    assert trace.termination == "finish"
    assert trace.tool_calls == 4
    assert len(trace.steps) == 6
    assert trace.steps[4]["validation_failed"] is True
    assert trace.steps[4]["verification"]["approved"] is False
    assert trace.steps[5]["verification"]["approved"] is True
    assert any(o.get("code") == "missing_or_duplicate_citations"
               for _question, observations, _left in policy.calls
               for o in observations)


def test_graph_step_and_tool_budgets_do_not_loop_forever():
    docs, q = corpus()
    repeated = [Action("search", query="Atlas7")] * 8
    a = GraphResearchAgent(docs, ScriptedPolicy(repeated),
                           AgentConfig(max_steps=2, max_tool_calls=7),
                           role_mode="single").run(q)
    assert a.termination == "step_budget_exhausted"
    assert a.tool_calls == 2
    b = GraphResearchAgent(docs, ScriptedPolicy(repeated),
                           AgentConfig(max_steps=5, max_tool_calls=1),
                           role_mode="single").run(q)
    assert b.termination == "tool_budget_exhausted"
    assert b.tool_calls == 1


def test_single_role_onpolicy_reward_remains_train_only():
    docs, q = corpus()
    runner = GraphResearchAgent(docs, FixtureMultiHopPolicy(), role_mode="single")
    trace = runner.run(q)
    assert trace.termination == "finish"
    assert all(step["role"] == "single_policy" for step in trace.steps)
    reward = score_train_rollout(q, trace, {d.doc_id: d for d in docs},
                                 max_tool_calls=7)
    assert reward.strict_outcome == 1.0
    assert reward.answer_correct == 1.0
    with pytest.raises(ValueError, match="TRAIN-only"):
        score_train_rollout(
            Question(q.question_id, q.question, q.gold_answer,
                     q.supporting_doc_ids, split="dev"),
            trace, {d.doc_id: d for d in docs},
        )


def test_sqlite_checkpoint_and_evaluator_manifest(tmp_path):
    docs, q = corpus()
    source = tmp_path / "data"
    source.mkdir()
    save_jsonl(source / "documents.jsonl", [vars(d) for d in docs])
    save_jsonl(source / "questions.jsonl", [
        {**vars(q), "supporting_doc_ids": list(q.supporting_doc_ids)}
    ])
    checkpoint = tmp_path / "private_graph_checkpoints.sqlite"
    output = tmp_path / "evaluation"
    result = evaluate(
        source, "train", FixtureMultiHopPolicy(), output,
        runtime="langgraph", graph_checkpoint_db=checkpoint,
        graph_role_mode="multi",
    )
    assert result["n"] == 1
    assert result["verified_success_mean"] == 1.0
    assert checkpoint.is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["agent_runtime"] == "langgraph"
    assert manifest["graph_checkpoint_backend"] == "sqlite"
    assert len(manifest["graph_checkpoint_sha256"]) == 64
    threads = [json.loads(line) for line in
               (output / "graph_threads.jsonl").read_text().splitlines()]
    assert threads[0]["question_id"] == q.question_id
    assert threads[0]["thread_id"].startswith("eval-42:")
    with pytest.raises(FileExistsError):
        evaluate(source, "train", FixtureMultiHopPolicy(), output,
                 runtime="langgraph")
    with pytest.raises(FileExistsError):
        evaluate(source, "train", FixtureMultiHopPolicy(), tmp_path / "new",
                 runtime="langgraph", graph_checkpoint_db=checkpoint)


def test_evidence_state_is_opt_in_and_nonoracle():
    docs, q = corpus()
    trace = GraphResearchAgent(
        docs, FixtureMultiHopPolicy(), evidence_state_mode="v3",
        min_citations=2,
    ).run(q)
    assert trace.termination == "finish"
    summary = trace.steps[-1]["evidence_state_before"]
    assert summary["state_version"] == "v3"
    assert "gold_answer" not in repr(summary)
    assert summary["warning"].endswith("NOT proven entailment")


def test_separate_planner_researcher_solver_policies_are_real_handoffs():
    docs, q = corpus()
    # Each logical agent owns its own policy object/model budget; they exchange
    # only the shared, gold-blind graph state, not the evaluator labels.
    planner = ScriptedPolicy([Action("search", query="Explorer Atlas7")])
    researcher = ScriptedPolicy([
        Action("read", doc_id="p"), Action("search", query="Region8"),
        Action("read", doc_id="r"),
    ])
    solver = ScriptedPolicy([
        Action("finish", answer="Quartz9", citations=("p", "r")),
    ])
    fallback = ScriptedPolicy([])
    agent = GraphResearchAgent(
        docs, fallback, role_mode="multi",
        role_policies={"planner": planner, "researcher": researcher, "solver": solver},
    )
    trace = agent.run(q)
    assert trace.termination == "finish"
    assert len(planner.calls) == 1
    assert len(researcher.calls) == 3
    assert len(solver.calls) == 1
    assert len(fallback.calls) == 0
    assert trace.policy.endswith("_role_split")
    assert set(trace.citations) == {"p", "r"}
    with pytest.raises(ValueError, match="require multi"):
        GraphResearchAgent(docs, fallback, role_mode="single",
                           role_policies={"planner": planner})
