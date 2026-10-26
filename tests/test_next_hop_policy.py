"""M7 frontier-first bridge planning: gold-blind and tool-budget tests."""
from sidra.agent import AgentConfig, SearchAgent
from sidra.next_hop_policy import (
    FrontierFirstBridgePolicy, _phrase_in, plan_next_hop,
)
from sidra.schema import Action, Document, Question


def observations(with_second=True):
    hits = [{"doc_id": "a", "title": "Ringling, Montana"}]
    if with_second:
        hits.append({"doc_id": "b", "title": "Chicago, Milwaukee, St. Paul and Pacific Railroad"})
    return [
        {"kind": "search", "query": "Which railroad served Ringling, Montana?",
         "hits": hits},
        {"kind": "read", "doc_id": "a", "title": "Ringling, Montana",
         "text": "Ringling, Montana was a station on the Chicago, Milwaukee, "
                 "St. Paul and Pacific Railroad. The line closed later."},
    ]


def test_reuse_visible_frontier_instead_of_pay_for_duplicate_search():
    result = plan_next_hop("Which railroad served Ringling, Montana?",
                           observations(), remaining_steps=3,
                           remaining_tool_calls=2)
    assert result is not None
    assert result.action == Action("read", doc_id="b")
    assert result.reason in {"frontier_exact_bridge_entity",
                             "frontier_title_mentioned_in_read"}
    assert result.action.query == ""


def test_when_bridge_is_not_visible_search_is_billed_normally():
    result = plan_next_hop("Which railroad served Ringling, Montana?",
                           observations(with_second=False), remaining_steps=3,
                           remaining_tool_calls=2)
    assert result is not None
    assert result.action.kind == "search"
    assert "Railroad" in result.action.query
    assert result.reason == "observed_second_hop_search"


def test_no_extra_search_if_budget_cannot_read_result_and_finish():
    assert plan_next_hop("Which railroad served Ringling, Montana?",
                         observations(with_second=False),
                         remaining_steps=2, remaining_tool_calls=2) is None
    assert plan_next_hop("Which railroad served Ringling, Montana?",
                         observations(with_second=False),
                         remaining_steps=3, remaining_tool_calls=1) is None
    # A visible already-exposed hit needs only one billed READ and FINISH.
    p = plan_next_hop("Which railroad served Ringling, Montana?",
                      observations(), remaining_steps=2,
                      remaining_tool_calls=1)
    assert p and p.action.kind == "read"


def test_never_propose_from_unobserved_hits_or_gold_labels():
    obs = observations(with_second=False)
    a = plan_next_hop("Which railroad served Ringling, Montana?", obs,
                      remaining_steps=4, remaining_tool_calls=3)
    b = plan_next_hop("Which railroad served Ringling, Montana?",
                      obs + [{"kind": "eval_only",
                              "gold_answer": "SECRET",
                              "supporting_doc_ids": ["secret-doc"],
                              "secret_context": ["Secret University"]}],
                      remaining_steps=4, remaining_tool_calls=3)
    assert a == b
    assert not any("SECRET" in str(x) for x in (a, b))
    assert plan_next_hop("Which railroad served Ringling, Montana?",
                         [{"kind": "read", "doc_id": "a",
                           "text": "Chicago Railroad"}],
                         remaining_steps=5, remaining_tool_calls=3) is None


def test_phrase_linkage_requires_whole_contiguous_tokens():
    assert _phrase_in("Harvard University", "The Harvard, University campus.")
    assert not _phrase_in("Harvard University", "The Harvardian University campus.")
    assert not _phrase_in("Art", "Earth and crafts")
    assert not _phrase_in("Brown University", "University of Brown")
    assert _phrase_in("Brown University", "Brown University established the centre.")


class Scripted:
    name = "scripted"
    model_calls = 0
    prompt_tokens = 0
    completion_tokens = 0

    def decide(self, question, observations, remaining):
        self.model_calls += 1
        if not observations:
            return Action("search", query="Ringling Montana")
        reads = [o for o in observations if o.get("kind") == "read"]
        if not reads:
            return Action("read", doc_id=observations[-1]["hits"][0]["doc_id"])
        return Action("finish", answer="Railroad",
                      citations=tuple(o["doc_id"] for o in reads))


def test_searchagent_uses_genuine_read_and_budget_with_no_gold_in_policy():
    docs = [
        Document("a", "Ringling Montana",
                 "Ringling Montana. Ringling Montana is a station served by "
                 "the Chicago Milwaukee St Paul and Pacific Railroad."),
        Document("b", "Chicago Milwaukee St Paul and Pacific Railroad",
                 "This railway was operated near Ringling Montana."),
    ]
    q = Question("sample", "Which railroad served Ringling Montana?",
                 gold_answer="NOT SHOWN TO POLICY", supporting_doc_ids=("a", "b"),
                 split="train")
    policy = FrontierFirstBridgePolicy(Scripted(), max_tool_calls=3)
    trace = SearchAgent(docs, policy, AgentConfig(max_steps=6,
               max_tool_calls=3, search_top_k=2)).run(q)
    assert trace.termination == "finish", trace.error
    assert trace.search_calls == 1
    assert trace.read_calls == 2
    assert trace.tool_calls == 3
    assert set(trace.citations) == {"a", "b"}
    reasons = [s.get("verification", {}).get("reason", "")
               for s in trace.steps]
    assert "frontier_title_mentioned_in_read" in reasons or (
        "frontier_exact_bridge_entity" in reasons)
    assert trace.model_calls == 3  # SEARCH, first READ, final FINISH


def test_invalid_policy_budget_rejected():
    import pytest
    with pytest.raises(ValueError):
        FrontierFirstBridgePolicy(Scripted(), max_tool_calls=0)


def test_reject_arbitrary_dev32_before_any_gold_is_read(tmp_path):
    import pytest
    from scripts.m7_retrieval_probe import run

    dataset = tmp_path / "not_an_open_pilot"
    dataset.mkdir()
    (dataset / "documents.jsonl").write_text('{"fake":"doc"}\n')
    (dataset / "questions.jsonl").write_text('{"fake":"question"}\n')
    with pytest.raises(ValueError, match="frozen open pilot32"):
        run(dataset, tmp_path / "output", mode="pilot")
    assert not (tmp_path / "output").exists()
