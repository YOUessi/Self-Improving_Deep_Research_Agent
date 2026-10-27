"""M7 E-025: source-removal NLI is a conservative, label-blind research gate."""
import pytest
from sidra.citation_counterfactual import citation_leave_one_out


class ConditionalNLI:
    def __init__(self,only_full=True,allow_question_only=False):
        self.calls=[]
        self.only_full=only_full
        self.allow_question_only=allow_question_only

    def score(self,premises,hypotheses):
        self.calls.append((list(premises),list(hypotheses)))
        rows=[]
        for p in premises:
            if self.allow_question_only:
                prob=0.96
            elif "Marie Curie:" in p and "Poland:" in p:
                prob=0.96
            elif not self.only_full and "Marie Curie:" in p:
                prob=0.96
            else:
                prob=0.08
            rows.append({
                "entailment":prob,
                "neutral":(1-prob)*.75,
                "contradiction":(1-prob)*.25,
            })
        return rows


def trace(*,citations=None):
    return {
        "question_id":"train","question":"Where was Marie Curie born?",
        "answer":"Warsaw","citations":citations or ["d1","d2"],
        "termination":"finish","steps":[
            {"action":{"kind":"search","query":"Marie Curie"},
             "observation":{"kind":"search","hits":[
                 {"doc_id":"d1"},{"doc_id":"d2"}]}},
            {"action":{"kind":"read","doc_id":"d1"},
             "observation":{"kind":"read","doc_id":"d1","title":"Marie Curie",
                            "text":"Marie Curie was born in Warsaw."}},
            {"action":{"kind":"read","doc_id":"d2"},
             "observation":{"kind":"read","doc_id":"d2","title":"Poland",
                            "text":"Warsaw is a city in Poland."}},
            {"action":{"kind":"finish","answer":"Warsaw",
                       "citations":citations or ["d1","d2"]}},
        ],
    }


def test_each_source_must_affect_the_nli_confidence():
    scorer=ConditionalNLI()
    t=trace()
    got=citation_leave_one_out(t,t["question"],scorer)
    assert got.status=="counterfactual_necessary_model_prediction"
    assert got.citation_ids==("d1","d2")
    assert got.additional_nli_calls==2
    assert got.weakest_entailment_drop==pytest.approx(.88)
    assert len(scorer.calls)==2
    assert len(scorer.calls[0][0])==1
    assert len(scorer.calls[1][0])==2
    assert "Marie Curie:" not in scorer.calls[1][0][0]
    assert "Poland:" not in scorer.calls[1][0][1]
    assert got.semantic_ground_truth=="not_established"


def test_high_confidence_single_document_shortcut_must_fail():
    scorer=ConditionalNLI(only_full=False)
    t=trace()
    got=citation_leave_one_out(t,t["question"],scorer)
    assert got.status=="blocked_evidence_nonessential"
    assert got.weakest_entailment_drop==pytest.approx(0.0)


def test_model_prior_cannot_justify_unnecessary_citations():
    scorer=ConditionalNLI(allow_question_only=True)
    t=trace()
    got=citation_leave_one_out(t,t["question"],scorer)
    assert got.status=="blocked_evidence_nonessential"


def test_missing_or_duplicate_citations_fail_before_any_nli_calls():
    scorer=ConditionalNLI()
    t=trace(citations=["d1","not_read"])
    assert citation_leave_one_out(t,t["question"],scorer).status=="not_eligible"
    assert not scorer.calls
    t=trace(citations=["d1","d1"])
    assert citation_leave_one_out(t,t["question"],scorer).status=="not_eligible"
    assert not scorer.calls


def test_untrusted_gold_keys_never_reach_language_model():
    scorer=ConditionalNLI()
    t=trace()
    t["gold_answer"]="secret-label"
    t["supporting_doc_ids"]=["gold-hidden-id"]
    t["original_supporting_facts"]=[["secret-original-sentence",0]]
    got=citation_leave_one_out(t,t["question"],scorer)
    assert got.gold_was_not_accessed is True
    payload=repr(scorer.calls)
    for forbidden in ("secret-label","gold-hidden-id","secret-original-sentence"):
        assert forbidden not in payload


def test_sentence_window_reallocation_fail_closed_without_ablation():
    scorer=ConditionalNLI(allow_question_only=True)
    t=trace()
    # A very large sentence in each document forces a 1450-character cap
    # when combined but not when considered separately.
    t["steps"][1]["observation"]["text"]="Warsaw "+"strange "*180+"end."
    t["steps"][2]["observation"]["text"]="Poland "+"strange "*180+"end."
    r=citation_leave_one_out(t,t["question"],scorer)
    # Because the original NLI baseline may be high on the full two-title
    # prompt, the ablation must not silently change source selection.
    assert r.status=="not_eligible"
    assert r.reason in ("sentence_window_budget_interaction",
                        "missing_source_sentence_window")
    assert len(scorer.calls)==1


def test_token_length_gate_fails_before_ablated_nli_calls():
    scorer=ConditionalNLI()
    class FakeTokenizer:
        def __call__(self,p,h,**kwargs):
            return {"input_ids":list(range(len(p)+len(h)))}
    scorer.tokenizer=FakeTokenizer()
    scorer.max_length=20
    t=trace()
    got=citation_leave_one_out(t,t["question"],scorer)
    assert got.status=="not_eligible"
    assert got.reason=="nli_input_would_truncate"
    assert got.model_input_tokens_max>scorer.max_length
    assert len(scorer.calls)==1


def test_invalid_threshold_rejected_without_model_calls():
    scorer=ConditionalNLI()
    with pytest.raises(ValueError,match="threshold"):
        citation_leave_one_out(trace(),"Where was Marie Curie born?",
                               scorer,removal_ceiling=.9)
    assert not scorer.calls
