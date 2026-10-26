"""Second-hop relation-tail rewrite: observed-only, zero budget increase."""
import pytest

from sidra.bridge_rewrite import relation_tail,rewrite_billed_second_search
from sidra.next_hop_policy import NextHopPlan
from sidra.schema import Action,Question
from scripts.m7_relation_tail_probe import (
    RelationTailProbePolicy,sample_train_cohorts,
)


def history():
    return [
        {"kind":"search","query":"The Duchy of Genoa territories",
         "hits":[{"doc_id":"a","title":"Duchy of Genoa"}]},
        {"kind":"read","doc_id":"a","title":"Duchy of Genoa",
         "text":"The Duchy of Genoa passed following the Congress of Vienna."},
    ]


def test_rewrite_only_legal_second_search_keeps_billing():
    question=("The Duchy of Genoa became a territory due to a meeting "
              "chaired by what Austrian statesman?")
    plan=NextHopPlan(Action("search",query="Congress of Vienna"),
                     "observed_second_hop_search",None)
    rewritten=rewrite_billed_second_search(plan,question,history())
    assert rewritten.action.kind=="search"
    assert rewritten.action.query=="Congress of Vienna austrian statesman"
    assert rewritten.action.doc_id==""
    assert "billed" in rewritten.reason
    assert rewritten.candidate is None


def test_no_fabricated_read_or_finished_answer():
    read=NextHopPlan(Action("read",doc_id="already_exposed"),
                     "visible_hit",None)
    assert rewrite_billed_second_search(read,"Who?",history()) is read
    assert rewrite_billed_second_search(None,"Who?",history()) is None
    plan=NextHopPlan(Action("search",query="Congress of Vienna"),"s",None)
    assert rewrite_billed_second_search(plan,"What?",history()) is plan
    with pytest.raises(ValueError,match="max_tokens"):
        relation_tail("What Austrian statesman?","Congress of Vienna",
                      "Duchy of Genoa",max_tokens=5)


def test_no_gold_or_unobserved_documents_can_affect_query():
    policy=RelationTailProbePolicy()
    question=("The Duchy of Genoa became a territory due to a meeting "
              "chaired by what Austrian statesman?")
    a=policy.decide(question,history(),remaining=4)
    # Offline fields never reach the policy, and even contaminated event kinds
    # must not modify its observed-only rewrite.
    b=policy.decide(question,history()+[{
        "kind":"eval_only","gold_answer":"SECRET",
        "supporting_doc_ids":["unseen"],"dev_answer":"HIDDEN"}],remaining=4)
    assert a==b
    assert a.kind=="search"
    assert "SECRET" not in a.query
    assert policy.rewrite_actions==2


def test_train_cohorts_disjoint_deterministic_never_dev():
    qs=[Question(f"id{i:04d}","a q","a",split="train") for i in range(200)]
    a=sample_train_cohorts(qs,n=64)
    b=sample_train_cohorts(list(reversed(qs)),n=64)
    assert {k:[q.question_id for q in v] for k,v in a.items()}=={
        k:[q.question_id for q in v] for k,v in b.items()}
    ids=[{q.question_id for q in v} for v in a.values()]
    assert not (ids[0]&ids[1])
    with pytest.raises(ValueError,match="TRAIN-only"):
        sample_train_cohorts([Question("d","a","a",split="dev")]+qs,n=64)


def test_any_unknown_pilot_dataset_is_rejected_before_loading(tmp_path):
    from scripts.m7_relation_tail_probe import run
    root=tmp_path/"fake"
    root.mkdir()
    (root/"documents.jsonl").write_text("{}\n")
    (root/"questions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError,match="frozen open pilot32"):
        run(root,tmp_path/"out",mode="pilot")
    assert not (tmp_path/"out").exists()
