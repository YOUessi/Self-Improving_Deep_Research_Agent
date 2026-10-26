"""Evidence-aware guard never sees gold and cannot invent document IDs."""
from sidra.agent import AgentConfig, SearchAgent
from sidra.grounded_policy import GroundedPolicy
from sidra.schema import Action, Document, Question


DOCS=[
    Document("doc-a","Alpha","Alpha belongs to Beta"),
    Document("doc-b","Beta","Beta belongs to Northern Province"),
    Document("doc-c","Unrelated","irrelevant"),
]


def test_unread_citations_trigger_real_read_not_fabricated_cite():
    class Scripted:
        name="scripted"
        def __init__(self):
            self.calls=0
            self.model_calls=0
            self.prompt_tokens=0
            self.completion_tokens=0
        def decide_with_system(self,question,observations,remaining,system):
            assert "gold_answer" not in system
            self.calls+=1
            self.model_calls+=1
            self.prompt_tokens+=10
            self.completion_tokens+=3
            if self.calls==1:
                return Action("search",query="Alpha Beta")
            if self.calls==2:
                return Action("read",doc_id="doc-a")
            return Action("finish",answer="Northern Province",
                          citations=("doc-a","doc-b"))
    wrapped=GroundedPolicy(Scripted(),min_reads=2,revise_citations=True)
    q=Question("q","Where does Alpha belong?", "Northern Province",
               supporting_doc_ids=("doc-a","doc-b"))
    t=SearchAgent(DOCS,wrapped,AgentConfig(search_top_k=3)).run(q)
    assert t.termination=="finish"
    assert [x["action"]["kind"] for x in t.steps]==["search","read","read","finish"]
    assert t.steps[2]["verification"]["reason"]=="unread_citation_repaired"
    assert t.citations==["doc-a","doc-b"]
    assert t.read_calls==2
    assert t.model_calls==4
    assert t.prompt_tokens==40


def test_duplicate_read_returns_unread_hit_without_gold_labels():
    class Duplicate:
        name="repeat"
        def decide(self,question,observations,remaining):
            return Action("read",doc_id="doc-a")
    p=GroundedPolicy(Duplicate(),min_reads=2)
    o=[
        {"kind":"search","query":"q","hits":[{"doc_id":"doc-a"},{"doc_id":"doc-b"}]},
        {"kind":"read","doc_id":"doc-a","text":"a"},
    ]
    a=p.decide("question",o,3)
    assert a.kind=="read" and a.doc_id=="doc-b"
    assert p.last_verification["reason"]=="duplicate_read_repaired"


def test_forced_read_happens_before_early_finish():
    class Early:
        name="early"
        def decide(self,question,observations,remaining):
            return Action("finish",answer="North",citations=("doc-a",))
    p=GroundedPolicy(Early(),min_reads=2)
    o=[
        {"kind":"search","query":"q","hits":[{"doc_id":"doc-a"},{"doc_id":"doc-b"}]},
        {"kind":"read","doc_id":"doc-a","text":"North"},
    ]
    result=p.decide("where",o,3)
    assert result.kind=="read" and result.doc_id=="doc-b"
    assert p.last_verification["reason"]=="finish_deferred_for_evidence"


def test_guard_does_not_pad_citations():
    class FailRevision:
        name="fail_rev"
        model_calls=0
        def decide_with_system(self,question,observations,remaining,system):
            self.model_calls+=1
            return Action("finish",answer="answer",citations=("doc-a",))
    base=FailRevision()
    p=GroundedPolicy(base,min_reads=2,revise_citations=True)
    o=[
        {"kind":"read","doc_id":"doc-a","text":"answer"},
        {"kind":"read","doc_id":"doc-b","text":"other"},
    ]
    a=p.decide("question",o,2)
    assert a.citations==("doc-a",)
    assert base.model_calls==2
    assert p.last_verification["reason"]=="citation_revision_unsuccessful"
