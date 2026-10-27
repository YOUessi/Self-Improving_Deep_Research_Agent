"""Unseen TRAIN64 is selected by question ID only, never answer labels."""
import pytest

from scripts.freeze_unseen_train64_for_hybrid import choose


def rows(n=10):
    return [{"id":f"train-{i}","answer":f"ANSWER-{i}",
             "supporting_facts":{"title":["a"],"sent_id":[0]}}
            for i in range(n)]


def test_exclusion_and_order_independent_of_answers():
    a=rows()
    excluded={"train-0","train-2","train-5"}
    first=choose(a,excluded,n=4)
    b=[dict(v,answer="CHANGED",supporting_facts={}) for v in a]
    second=choose(b,excluded,n=4)
    assert [x["id"] for x in first]==[x["id"] for x in second]
    assert {x["id"] for x in first}.isdisjoint(excluded)


def test_no_replacement_and_no_silent_shortfall():
    with pytest.raises(ValueError,match="not enough"):
        choose(rows(5),{"train-1","train-2","train-3","train-4"},n=3)
    with pytest.raises(ValueError,match="duplicate"):
        choose([rows(1)[0],rows(1)[0]],set(),n=2)
    with pytest.raises(ValueError,match="nonempty"):
        choose(rows(3),set(),n=0)
