"""Bridge proposer may use public question and actually observed READ only."""
from sidra.bridge import phrases, next_bridge_query, bridge_candidates


def test_bridge_from_read_document_named_entity():
    obs=[
        {"kind":"search","query":"railroad stop in Ringling Montana",
         "hits":[{"doc_id":"ringling","title":"Ringling, Montana"}]},
        {"kind":"read","doc_id":"ringling","title":"Ringling, Montana",
         "text":"Ringling, Montana was a station on the Chicago, Milwaukee, St. Paul and Pacific Railroad. The line was dismantled."},
    ]
    candidate=next_bridge_query("When did the railroad stop in Ringling, Montana operate?",obs)
    assert candidate is not None
    assert candidate.source in {"question_other_entity","read_entity"}
    assert "railroad" in candidate.query.lower() or "ringling" not in candidate.query.lower()


def test_new_candidates_do_not_include_evaluator_labels():
    question="Which railroad served Ringling, Montana?"
    obs=[{"kind":"read","doc_id":"ringling","title":"Ringling, Montana",
          "text":"A town served by the Chicago, Milwaukee, St. Paul and Pacific Railroad."}]
    a=next_bridge_query(question,obs)
    b=next_bridge_query(question,obs+[{"kind":"eval_only",
                                     "gold_answer":"SECRET",
                                     "supporting_doc_ids":["secret-id"]}])
    assert a==b


def test_empty_without_read_observation():
    assert next_bridge_query("Question",[{"kind":"search","query":"Q","hits":[]}]) is None


def test_prior_queries_not_repeated():
    candidates=bridge_candidates("Who works at Brown University?","Berkeley",
                  "The campus of Brown University is in Providence.",
                  prior_queries={"Brown University"})
    assert all(c.query!="Brown University" for c in candidates)
