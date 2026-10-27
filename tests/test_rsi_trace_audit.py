"""Recorded tool responses are checked against deterministic SearchAgent replay."""
from copy import deepcopy

from sidra.agent import AgentConfig, SearchAgent
from sidra.rsi_trace_audit import audit_tool_trace
from sidra.schema import Action,AgentTrace,Document,Question


def fixture():
    docs=[
        Document("a","Mason Bridge","Mason Bridge stands in Port"),
        Document("b","Port","Port is an inland city near Mason Bridge"),
        Document("c","Inland City","Distant inland city far away"),
    ]
    q=Question("q-train","What city is home to the Mason Bridge?","Port",
               supporting_doc_ids=("a","b"),split="train")
    actions=[
        Action("search",query="Mason Bridge Port"),
        Action("read",doc_id="a"),
        Action("read",doc_id="b"),
        Action("finish",answer="Port",citations=("a","b")),
    ]
    class RecordedPolicy:
        name="fixture_model_not_real_training"
        def __init__(self):self.idx=0
        def decide(self,question,observations,remaining):
            a=actions[self.idx]
            self.idx+=1
            return a
    cfg=AgentConfig(max_steps=4,max_tool_calls=3,search_top_k=3)
    trace=SearchAgent(docs,RecordedPolicy(),cfg).run(q)
    assert trace.termination=="finish"
    return docs,q,cfg,trace


def audited(docs,q,cfg,t):
    return audit_tool_trace(q,t,docs,config=cfg)


def test_valid_real_tool_executor_trace_passes():
    docs,q,cfg,t=fixture()
    verdict=audited(docs,q,cfg,t)
    assert verdict.structurally_valid
    assert verdict.complete_legal_finish
    assert verdict.checked_searches==1
    assert verdict.checked_reads==2
    assert verdict.checked_tool_calls==3


def test_search_hit_forgery_fails():
    docs,q,cfg,t=fixture()
    z=deepcopy(t)
    z.steps[0]["observation"]["hits"][0]["snippet"]="unobserved fabricated evidence"
    verdict=audited(docs,q,cfg,z)
    assert not verdict.structurally_valid
    assert verdict.reason=="search_hit_forgery"


def test_score_forgery_fails():
    docs,q,cfg,t=fixture()
    z=deepcopy(t)
    z.steps[0]["observation"]["hits"][0]["score"]+=2
    assert audited(docs,q,cfg,z).reason=="search_rank_or_score_forgery"


def test_read_text_forgery_fails():
    docs,q,cfg,t=fixture()
    z=deepcopy(t)
    z.steps[2]["observation"]["text"]="Fake Port was proven by the model"
    assert audited(docs,q,cfg,z).reason=="read_observation_forgery"


def test_finish_citation_did_not_read_fails():
    docs,q,cfg,t=fixture()
    z=deepcopy(t)
    z.steps[3]["action"]["citations"]=["a","c"]
    z.citations=["a","c"]
    assert audited(docs,q,cfg,z).reason=="finish_cites_unread_source"


def test_budget_or_counter_forgery_fails():
    docs,q,cfg,t=fixture()
    z=deepcopy(t)
    z.read_calls=100
    assert audited(docs,q,cfg,z).reason=="tool_counter_mismatch"
    z=deepcopy(t)
    z.steps[0]["step"]=4
    assert audited(docs,q,cfg,z).reason=="step_index_or_shape_mismatch"


def test_gold_blind_audit_rejects_dev_question():
    docs,q,cfg,t=fixture()
    q=Question(q.question_id,q.question,q.gold_answer,
               supporting_doc_ids=q.supporting_doc_ids,split="dev")
    assert audited(docs,q,cfg,t).reason=="train_only_verifier"


def test_no_citation_does_not_count_as_positive():
    docs,q,cfg,t=fixture()
    z=deepcopy(t)
    z.steps[-1]["action"]["citations"]=[]
    z.citations=[]
    v=audited(docs,q,cfg,z)
    assert v.structurally_valid
    assert not v.complete_legal_finish


def test_empty_finish_is_legal_but_not_positive():
    docs,q,cfg,_=fixture()
    class Abstaining:
        name="abstain"
        def decide(self,question,observations,remaining):
            return Action("finish")
    t=SearchAgent(docs,Abstaining(),cfg).run(q)
    v=audited(docs,q,cfg,t)
    assert v.structurally_valid and not v.complete_legal_finish
    assert v.reason=="nonpositive_abstained"
