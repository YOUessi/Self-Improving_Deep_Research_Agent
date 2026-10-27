"""M8 V3: one observable chain for real SEARCH/READ/FINISH decisions.

No original gold answer, support IDs or evaluator verdict enters the state.
These are mechanism tests, NOT an improved QA benchmark result.
"""
import json
from types import SimpleNamespace
import math

import pytest

from sidra.evidence_state import derive_evidence_state
from sidra.finite_action import (FiniteActionConfig,FiniteActionPolicy,
                                  build_legal_actions,action_prompt,
                                  logprob_of_action)
from sidra.schema import Action


Q="Which province contains River City?"
SEARCH={"kind":"search","query":Q,"hits":[
    {"doc_id":"a","title":"River City","snippet":"A historic town"},
    {"doc_id":"d","title":"Unrelated Castle","snippet":"A museum with Pine Oak"},
    {"doc_id":"b","title":"Coastal Province","snippet":"A geographic province"},
]}
SEARCH_NO_BRIDGE={"kind":"search","query":Q,"hits":[
    {"doc_id":"a","title":"River City","snippet":"A historic town"},
    {"doc_id":"d","title":"Unrelated Castle","snippet":"A museum"},
]}
READ_A={"kind":"read","doc_id":"a","title":"River City",
        "text":"River City is located in Coastal Province. River City was founded in 1800."}
READ_B={"kind":"read","doc_id":"b","title":"Coastal Province",
        "text":"Coastal Province contains River City."}
READ_D={"kind":"read","doc_id":"d","title":"Unrelated Castle",
        "text":"Unrelated Castle houses Pine Oak, a valuable painting."}


def test_frontier_link_is_a_real_observed_pending_read_not_hidden_title():
    s0=derive_evidence_state(Q,[SEARCH],version="v3")
    s1=derive_evidence_state(Q,[SEARCH,READ_A],version="v3")
    s2=derive_evidence_state(Q,[SEARCH,READ_A,READ_D],version="v3")
    s3=derive_evidence_state(Q,[SEARCH,READ_A,READ_D,READ_B],version="v3")
    assert s0.potential==pytest.approx(0.1)
    assert s1.pending_linked_hit_ids==("b",)
    assert s1.evidence_gap=="need_linked_read"
    assert s1.next_bridge_query is None  # READ exposed hit, no duplicate SEARCH
    assert s2.potential==s1.potential  # unrelated READ is NOT progress
    assert s3.anchored_linked_pairs==(("a","b"),)
    assert s3.evidence_gap=="linked_sources_answer_check"
    assert s3.potential>s2.potential>s0.potential
    assert s3.chain_candidate_support("Coastal Province",("a","b","d"))[0]>=1
    assert s3.chain_candidate_support("Pine Oak",("a","b","d"))==(0,0)
    assert s3.chain_candidate_support("Coastal Province",("d",))==(0,0)
    assert "gold_answer" not in json.dumps(s3.public_summary())


def test_irrelevant_last_read_cannot_hijack_anchored_bridge_search():
    s=derive_evidence_state(Q,[SEARCH_NO_BRIDGE,READ_A,READ_D],version="v3")
    assert s.anchored_read_ids==("a",)
    assert s.pending_linked_hit_ids==()
    assert s.next_bridge_query is not None
    assert "Coastal Province".casefold()==s.next_bridge_query.casefold()
    actions=build_legal_actions(Q,[SEARCH_NO_BRIDGE,READ_A,READ_D],
        remaining=3,max_tool_calls=5,
        cfg=FiniteActionConfig(answer_proposer="v4",evidence_state_mode="v3"))
    assert any(a.kind=="search" and a.query.casefold()=="coastal province"
               for a in actions)


def test_linked_read_and_chain_aware_finish_both_win_the_same_catalog():
    cfg=FiniteActionConfig(answer_proposer="v4",evidence_state_mode="v3")
    actions=build_legal_actions(Q,[SEARCH,READ_A],remaining=3,
                                max_tool_calls=5,cfg=cfg)
    reads=[a for a in actions if a.kind=="read"]
    assert reads and reads[0].doc_id=="b"
    state=derive_evidence_state(Q,[SEARCH,READ_A,READ_B,READ_D],version="v3")
    good=Action("finish",answer="Coastal Province",citations=("a","b","d"))
    distractor=Action("finish",answer="Pine Oak",citations=("a","b","d"))
    assert state.action_biases([good])[0]>state.action_biases([distractor])[0]
    finish=build_legal_actions(Q,[SEARCH,READ_A,READ_B,READ_D],
                               remaining=1,max_tool_calls=5,cfg=cfg)
    assert all(a.kind=="finish" for a in finish)
    assert any("Coastal Province".casefold() in a.answer.casefold()
               for a in finish if a.answer)
    assert len(finish)<=cfg.max_candidates


class SpyTokenizer:
    def __init__(self):self.content=""
    def apply_chat_template(self,messages,**kw):
        self.content=messages[-1]["content"]
        return self.content
    def __call__(self,s,add_special_tokens=False):return {"input_ids":[1,2,3]}
    def encode(self,s,add_special_tokens=False):
        return [20+int(s)] if len(s)==1 and s.isdigit() else [1,2]


def test_prompt_v3_is_chain_visible_and_gold_blind():
    tok=SpyTokenizer()
    cfg=FiniteActionConfig(answer_proposer="v4",evidence_state_mode="v3")
    obs=[SEARCH,READ_A]
    acts=build_legal_actions(Q,obs,cfg=cfg,remaining=3,max_tool_calls=5)
    assert action_prompt(tok,Q,obs,3,acts,cfg=cfg)
    msg=json.loads(tok.content)
    state=msg["evidence_state"]
    assert state["version"]=="v3"
    assert state["next_evidence_gap"]=="need_linked_read"
    assert state["pending_linked_read_ids"]==["b"]
    assert "gold_answer" not in tok.content and "supporting_doc_ids" not in tok.content
    assert "Pine Oak" not in str(state)


def test_old_versions_still_have_the_same_frozen_behavior():
    assert derive_evidence_state(Q,[SEARCH,READ_A,READ_D],version="v2").version=="v2"
    assert derive_evidence_state(Q,[SEARCH,READ_A,READ_D],version="v1").version=="v1"
    with pytest.raises(ValueError,match="requires V4/V5"):
        FiniteActionConfig(answer_proposer="legacy",evidence_state_mode="v3")


def test_v3_uses_exact_same_masked_distribution_for_sampling_and_training():
    torch=pytest.importorskip("torch")
    class ToyModel:
        def __init__(self):
            self.device=torch.device("cpu")
            self.weight=torch.nn.Parameter(torch.tensor(0.23))
        def eval(self):return self
        def __call__(self,input_ids):
            logits=torch.arange(40,dtype=torch.float32)*self.weight
            return SimpleNamespace(logits=logits.reshape(1,1,40).expand(
                1,input_ids.shape[-1],40))
    tok=SpyTokenizer()
    pol=FiniteActionPolicy(ToyModel(),tok,seed=43,
        cfg=FiniteActionConfig(answer_proposer="v4",evidence_state_mode="v3"),
        max_tool_calls=5)
    action=pol.decide(Q,[SEARCH,READ_A,READ_B],remaining=1)
    sample=pol.samples[-1]
    assert sample["evidence_state_mode"]=="v3"
    assert sample["evidence_state"]["linked_chain_ready"] is True
    logps=torch.stack([logprob_of_action(
        pol.model,sample["prompt_ids"],sample["legal_choice_token_ids"],i,
        logit_biases=sample["action_logit_biases"])
        for i in range(sample["num_legal_actions"])])
    assert math.isclose(float(logps.exp().sum()),1.,abs_tol=1e-6)
    assert float(logps[sample["selected_index"]])==pytest.approx(
        sample["sampled_logprob"],abs=1e-6)
    assert action.to_dict() in sample["legal_actions"]
