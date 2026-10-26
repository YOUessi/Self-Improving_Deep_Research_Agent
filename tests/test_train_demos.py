"""Training converter preserves label containment and teacher provenance."""
import json
import pytest

from scripts.build_train_demos import to_hotpot_original, TrainOnlyOraclePolicy
from sidra.schema import Action


def test_hf_parquet_struct_to_original_json_format():
    row={"id":"train001","question":"Which region?", "answer":"North",
         "context":{"title":["A","B"],"sentences":[["A links B."],["B is in North."]]},
         "supporting_facts":{"title":["A","B"],"sent_id":[0,0]},
         "type":"bridge","level":"easy"}
    converted=to_hotpot_original(row)
    assert converted["_id"]=="train001"
    assert converted["context"]==[["A",["A links B."]],["B",["B is in North."]]]
    assert converted["supporting_facts"]==[["A",0],["B",0]]


def test_training_policy_is_explicitly_oracle_not_llm():
    items=[Action("search",query="q"),Action("read",doc_id="a"),
           Action("finish",answer="North",citations=("a","b"))]
    teacher=TrainOnlyOraclePolicy(items)
    assert "oracle" in teacher.name
    assert [teacher.decide("q",[],i) for i in (3,2,1)]==items
    assert teacher.decide("q",[],0).answer==""
    assert not hasattr(teacher,"model_calls")


def test_malformed_hotpot_nested_support_rejected():
    with pytest.raises(ValueError,match="supporting-fact"):
        to_hotpot_original({"id":"x","question":"?","answer":"A",
              "context":{"title":["A"],"sentences":[["A"]]},
              "supporting_facts":{"title":["A"],"sent_id":[]}})
