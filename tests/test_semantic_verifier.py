"""M7 E-024: adversarial and trust-domain tests of CPU-only NLI gate."""
import pytest

from sidra.semantic_verifier import (
    question_answer_hypothesis, actual_cited_reads,
    sentence_windows, semantic_review,observed_cross_citation_title_link,
)


class ToyNLI:
    def __init__(self, scores):
        self.probs=scores
        self.calls=[]
    def score(self,premises,hypotheses):
        self.calls.append((premises,hypotheses))
        return [self.probs for _ in hypotheses]


def trace(*,premise="Marie Curie was born in Warsaw in 1867.",
          answer="Warsaw",qid="q1",invalid_cite=False):
    citations=["d1","d2"] if not invalid_cite else ["d1","d3"]
    return {
        "question_id":qid,
        "question":"Where was Marie Curie born?",
        "answer":answer,"citations":citations,"termination":"finish",
        "steps":[
            {"action":{"kind":"search","query":"Marie Curie"},
             "observation":{"kind":"search","hits":[
                 {"doc_id":"d1","title":"Marie Curie"},
                 {"doc_id":"d2","title":"Poland"}]}},
            {"action":{"kind":"read","doc_id":"d1"},
             "observation":{"kind":"read","doc_id":"d1","title":"Marie Curie",
                            "text":premise}},
            {"action":{"kind":"read","doc_id":"d2"},
             "observation":{"kind":"read","doc_id":"d2","title":"Poland",
                            "text":"Warsaw is in Poland."}},
            {"action":{"kind":"finish","answer":answer,"citations":citations}},
        ],
    }


def test_question_to_claim_abstains_on_ambiguous_bridge():
    assert question_answer_hypothesis(
        "Where was Marie Curie born?", "Warsaw")=="Marie Curie was born in Warsaw."
    assert question_answer_hypothesis(
        "What author created the fictional character?", "Alice"
    )=="Alice created the fictional character."
    assert question_answer_hypothesis(
        "In what year was the university founded?", "1950"
    )=="The university was founded in 1950."
    assert question_answer_hypothesis(
        "The Duke was a son of what king?", "Henry VII") is None
    assert question_answer_hypothesis("What?", "") is None
    assert question_answer_hypothesis(
        "Rock'n Rollers were a subsidiary of what company since 2006?",
        "Tayto")=="Rock'n Rollers were a subsidiary of Tayto since 2006."
    assert question_answer_hypothesis(
        "Darin Adler was a developer of the engine that ran which point-and-click adventure?",
        "MacVenture")=="Darin Adler was a developer of the engine that ran MacVenture."
    assert question_answer_hypothesis(
        "A reality show features what two parents of the Duggar family?",
        "Jim Bob and Michelle Duggar"
    )=="A reality show features Jim Bob and Michelle Duggar of the Duggar family."
    assert question_answer_hypothesis(
        "The Latino World Order was led by a wrestler born in which year?",
        "1962"
    )=="The Latino World Order was led by a wrestler born in 1962."
    assert question_answer_hypothesis(
        "Which of these films won what award?", "Palme d'Or") is None


def test_positive_nli_is_model_prediction_never_oracle_truth():
    scorer=ToyNLI({"entailment":0.94,"neutral":0.04,"contradiction":0.02})
    t=trace()
    v=semantic_review(t,t["question"],scorer)
    assert v.status=="model_predicted_entailment"
    assert v.semantic_ground_truth=="not_established"
    assert v.hypothesis=="Marie Curie was born in Warsaw."
    assert v.cited_read_ids==("d1","d2")
    assert "born in Warsaw" in scorer.calls[0][0][0]
    assert v.entailment==pytest.approx(0.94)


def test_counterexample_answer_present_in_NEGATED_SOURCE_not_trusted():
    # Lexical passage proof incorrectly accepts "Warsaw" in a negation;
    # model-based NLI can flag a contradiction, but still only as prediction.
    scorer=ToyNLI({"entailment":0.01,"neutral":0.02,"contradiction":0.97})
    t=trace(premise="Marie Curie was not born in Warsaw.")
    v=semantic_review(t,t["question"],scorer)
    assert v.status=="model_predicted_contradiction"
    assert v.semantic_ground_truth=="not_established"


def test_unread_citations_error_or_unsupported_question_cannot_call_nli():
    scorer=ToyNLI({"entailment":0.99,"neutral":0.005,"contradiction":0.005})
    t=trace(invalid_cite=True)
    assert semantic_review(t,t["question"],scorer).status=="not_scored"
    assert scorer.calls==[]
    t=trace()
    assert semantic_review(t,"Which of these two novels is longer?",scorer).reason=="unsupported_question_form"
    assert scorer.calls==[]


def test_search_snippets_and_wrong_doc_action_never_enter_nli():
    t=trace()
    t["steps"][1]["action"]["doc_id"]="FAKE"
    r=actual_cited_reads(t)
    assert "d1" not in r
    scorer=ToyNLI({"entailment":0.99,"neutral":0.005,"contradiction":0.005})
    verdict=semantic_review(t,t["question"],scorer)
    assert verdict.status=="not_scored"
    assert scorer.calls==[]


def test_no_cross_document_phantom_answer_nor_unread_gold():
    scorer=ToyNLI({"entailment":0.99,"neutral":0.005,"contradiction":0.005})
    t=trace(answer="New York",
        premise="Marie Curie was born in New")
    t["steps"][2]["observation"]["text"]="York is the name of a town."
    verdict=semantic_review(t,t["question"],scorer)
    assert verdict.reason=="lexical_answer_not_in_any_cited_read"
    assert scorer.calls==[]


def test_sentence_extraction_reads_cited_docs_only_deterministically():
    reads={"a":{"title":"One","text":"Unrelated item. Marie Curie was born in Warsaw."},
           "b":{"title":"Two","text":"Warsaw was in Poland. Nobody mentioned Mars."},
           "hidden":{"title":"Hidden","text":"The answer was London."}}
    premise,n=sentence_windows("Where was Marie Curie born?","Warsaw",
                               reads,("a","b"),max_chars=300)
    assert n==4
    assert "born in Warsaw" in premise and "Hidden" not in premise
    assert premise==sentence_windows("Where was Marie Curie born?","Warsaw",
                                    reads,("a","b"),max_chars=300)[0]


def test_nli_probabilistic_consistency_fail_closed():
    scorer=ToyNLI({"entailment":0.88,"neutral":0.07,"wrong":0.05})
    with pytest.raises(ValueError,match="label map"):
        semantic_review(trace(),"Where was Marie Curie born?",scorer)
    scorer.probs={"entailment":3.0,"neutral":-2.0,"contradiction":0.0}
    with pytest.raises(ValueError,match="posterior"):
        semantic_review(trace(),"Where was Marie Curie born?",scorer)


def test_same_trace_no_gold_fields_are_passed_to_nli():
    scorer=ToyNLI({"entailment":0.92,"neutral":0.07,"contradiction":0.01})
    t=trace()
    t["gold_answer"]="Berlin"
    t["supporting_doc_ids"]=["secret-id"]
    t["original_supporting_facts"]=[["Secret",1]]
    v=semantic_review(t,t["question"],scorer)
    assert v.status=="model_predicted_entailment"
    serialized=str(scorer.calls)
    assert "Berlin" not in serialized
    assert "secret-id" not in serialized
    assert "Secret" not in serialized


def test_observed_title_link_is_sourced_from_distinct_cited_reads_only():
    docs={
        "crisp":{"title":"Rock 'n' Roller Crisps",
                 "text":"Rock'n Rollers were crisps made by Golden Wonder."},
        "maker":{"title":"Golden Wonder",
                 "text":"Golden Wonder has been owned by Tayto since 2006."},
        "hidden":{"title":"Secret","text":"Unknown"},
    }
    assert observed_cross_citation_title_link(docs,("crisp","maker"))
    assert not observed_cross_citation_title_link(docs,("maker","hidden"))
    assert not observed_cross_citation_title_link(docs,("crisp",))
    assert not observed_cross_citation_title_link(docs,("crisp","maker","hidden"))


def test_unlinked_high_confidence_nli_is_rejected_but_prediction_retained():
    scorer=ToyNLI({"entailment":0.995,"neutral":0.004,"contradiction":0.001})
    t=trace()
    assert semantic_review(t,t["question"],scorer).status=="model_predicted_entailment"
    guarded=semantic_review(t,t["question"],scorer,require_bridge_link=True)
    assert guarded.status=="model_entailment_blocked_no_observed_bridge"
    assert guarded.observed_bridge_link is False
    assert guarded.model_prediction=="entailment"
    # Now establish an actual cross-READ link using one *observed* title.
    t["steps"][2]["observation"]["text"]="Marie Curie was born in Warsaw, Poland."
    accepted=semantic_review(t,t["question"],scorer,require_bridge_link=True)
    assert accepted.status=="model_predicted_entailment"
    assert accepted.observed_bridge_link is True


def test_partial_title_overlap_is_not_cross_citation_link():
    reads={
        "a":{"title":"The Latino World Order",
             "text":"Eddie Guerrero led the Latino World Order."},
        "b":{"title":"Jeff Farmer (wrestler)",
             "text":"Jeff Farmer was born in 1962 and played the New World Order persona."},
    }
    assert not observed_cross_citation_title_link(reads,("a","b"))
    accented={
        "a":{"title":"Acis et Galatée",
             "text":"The work belongs to the pastorale-héroïque genre."},
        "b":{"title":"Pastorale héroïque",
             "text":"The first work was Acis et Galatée by Lully."},
    }
    assert observed_cross_citation_title_link(accented,("a","b"))
