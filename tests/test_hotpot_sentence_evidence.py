"""Original supporting-sentence evaluator is offline and does not leak labels."""
import json
import pytest
from scripts.audit_hotpot_sentence_evidence import audit,audit_case


ORIGINAL={
 "_id":"dev-1",
 "question":"Who lives in River City?",
 "answer":"Mara",
 "context":[
   ["A",["Mara was born in City B.","Mara lives in River City."]],
   ["B",["River City is a location.","River City is on Island X."]],
 ],
 "supporting_facts":[["A",1],["B",0]],
}


def trace(*,title_b="B",text_b="River City is a location.",cite_b=True):
    s=[
     {"observation":{"kind":"read","doc_id":"a","title":"A",
                     "text":"Mara lives in River City."}},
     {"observation":{"kind":"read","doc_id":"b","title":title_b,
                     "text":text_b}},
    ]
    return {"question_id":"dev-1","answer":"Mara",
            "citations":["a","b"] if cite_b else ["a"],
            "steps":s}


def test_sentence_presence_requires_both_gold_sentences_in_actual_read_citations():
    good=audit_case(ORIGINAL,trace())
    assert good["answer_exact"]==1
    assert good["sentence_support_read_complete"]
    assert good["sentence_support_cited_complete"]
    assert good["answer_and_sentence_support_complete"]
    assert good["invalid_original_sentence_indices"]==0
    without_sentence=audit_case(ORIGINAL,trace(text_b="Only a different sentence"))
    assert without_sentence["passage_support_complete"]
    assert not without_sentence["sentence_support_cited_complete"]
    assert not without_sentence["answer_and_sentence_support_complete"]
    with_wrong_cite=audit_case(ORIGINAL,trace(cite_b=False))
    assert with_wrong_cite["sentence_support_read_complete"]
    assert not with_wrong_cite["sentence_support_cited_complete"]


def test_heldout_summary_has_no_case_level_gold(tmp_path):
    data=tmp_path/"dev_original.json"
    data.write_text(json.dumps([ORIGINAL]))
    tr=tmp_path/"traces.jsonl"
    tr.write_text(json.dumps(trace())+"\n")
    out=tmp_path/"report"
    summary=audit(data,tr,out,heldout=True)
    assert summary["n"]==1
    assert summary["answer_and_sentence_support_complete_count"]==1
    assert not (out/"pilot_per_case.jsonl").exists()
    with pytest.raises(FileExistsError):audit(data,tr,out,heldout=True)
    tr.write_text(json.dumps({**trace(),"question_id":"unknown"})+"\n")
    with pytest.raises(ValueError,match="frozen"):
        audit(data,tr,tmp_path/"invalid",heldout=True)
