"""Bridge policy is bounded, gold-blind and uses original search/read tool."""
from sidra.bridge_policy import BridgeGroundedPolicy,choose_unread_hit
from sidra.schema import Action


class Dummy:
    name="grounded_test"
    model_name="my-model"
    model_calls=0
    prompt_tokens=0
    completion_tokens=0
    def decide(self,question,observations,remaining):
        self.model_calls+=1
        if not observations:return Action("search",query="starting question")
        if observations[-1]["kind"]=="search":
            return Action("read",doc_id=observations[-1]["hits"][0]["doc_id"])
        return Action("finish",answer="answer",citations=("a",))


def test_bridge_after_real_read_and_before_finish():
    p=BridgeGroundedPolicy(Dummy())
    o=[
        {"kind":"search","query":"station stop at Ringling Montana",
         "hits":[{"doc_id":"a","title":"Ringling, Montana"}]},
        {"kind":"read","doc_id":"a","title":"Ringling, Montana",
         "text":"The station was served by the Chicago, Milwaukee, St. Paul and Pacific Railroad."}
    ]
    a=p.decide("When did the railroad in Ringling, Montana operate?",o,remaining=4)
    assert a.kind=="search"
    assert p.model_calls==0
    assert p.last_verification["reason"]=="gold_blind_second_hop_search"
    assert p.last_role=="bridge_planner"
    # Must never add an invisible doc ID or use a gold support label.
    b=p.decide("When?",o+[{"kind":"search","query":a.query,
                            "hits":[{"doc_id":"b","title":"Chicago, Milwaukee, St. Paul and Pacific Railroad"}]}],remaining=3)
    assert b.kind=="read" and b.doc_id=="b"
    assert p.last_verification["reason"]=="bridge_observed_frontier_read"


def test_bridge_does_not_bypass_remaining_budget():
    p=BridgeGroundedPolicy(Dummy())
    o=[{"kind":"read","title":"Brown University",
        "doc_id":"a","text":"Harvard University is located in Cambridge."}]
    a=p.decide("Where is Brown University?",o,remaining=2)
    assert a.kind=="finish"
    assert p.model_calls==1


def test_choose_unread_honors_visibility_not_gold():
    o=[
        {"kind":"search","query":"A","hits":[{"doc_id":"a","title":"A"}]},
        {"kind":"read","doc_id":"a","title":"A","text":"Visit Brown University"},
        {"kind":"search","query":"Brown University",
         "hits":[{"doc_id":"a","title":"A"},{"doc_id":"b","title":"Brown University"}]},
    ]
    assert choose_unread_hit("Where is Brown University?",o)=="b"
    assert choose_unread_hit("Where is Brown University?",o+[{
        "kind":"eval_only","gold_answer":"Secret",
        "supporting_doc_ids":["secret"]}])=="b"
