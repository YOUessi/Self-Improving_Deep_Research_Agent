"""Evidence-first v3 policy: no-oracle guards and TRAIN task curriculum."""
import pytest
from sidra.finite_action import FiniteActionConfig,build_legal_actions
from sidra.schema import Question
from scripts.train_finite_action_grpo import select_train_only


def _observations():
    return [
        {"kind":"search","query":"Congress of Vienna leaders",
         "hits":[
             {"doc_id":"s::1","title":"Vienna summit","snippet":"Chaired by Klemens"},
             {"doc_id":"s::2","title":"Klemens","snippet":"Austrian statesman"},
             {"doc_id":"s::3","title":"Europe","snippet":"Europe history"},
             {"doc_id":"s::4","title":"Background","snippet":"Vienna"},
         ]},
        {"kind":"read","doc_id":"s::1","title":"Vienna summit",
         "text":"The Congress of Vienna was chaired by Klemens von Metternich in 1815. "
                "Klemens von Metternich led the diplomatic negotiations."},
        {"kind":"read","doc_id":"s::2","title":"Klemens",
         "text":"Austrian statesman Klemens von Metternich was born in 1773 and "
                "served under Emperor Francis II."},
    ]


def test_v3_requires_two_visible_distinct_reads_before_factual_finish():
    q="Who chaired the Congress of Vienna?"
    c=FiniteActionConfig(answer_proposer="v3")
    assert not any(a.kind=="finish" and a.answer
                   for a in build_legal_actions(q,[],cfg=c,remaining=6))
    one=_observations()[:2]
    v2=build_legal_actions(q,one,
                           cfg=FiniteActionConfig(answer_proposer="v2"),
                           remaining=4,max_tool_calls=5)
    v3=build_legal_actions(q,one,cfg=c,remaining=4,max_tool_calls=5)
    assert any(a.kind=="finish" and a.answer for a in v2)
    assert not any(a.kind=="finish" and a.answer for a in v3)
    assert any(a.kind=="read" and a.doc_id=="s::2" for a in v3)
    assert v3[-1].kind=="finish" and v3[-1].answer==""
    assert all(not a.citations for a in v3)


def test_v3_preserves_correct_candidate_slots_and_complete_visible_citations():
    q="Which Austrian statesman chaired the Congress of Vienna?"
    obs=_observations()
    cfg=FiniteActionConfig(answer_proposer="v3")
    actions=build_legal_actions(q,obs,cfg=cfg,remaining=3,max_tool_calls=5)
    answers=[a for a in actions if a.kind=="finish" and a.answer]
    assert answers, "Must expose at least one nonoracle candidate after two reads"
    assert len(actions)<=cfg.max_candidates
    assert actions[-1].kind=="finish" and actions[-1].answer==""
    assert all(a.citations==("s::1","s::2") for a in answers)
    assert any(a.kind=="search" for a in actions)
    assert any(a.kind=="read" for a in actions)
    assert any("Metternich" in a.answer for a in answers)
    assert all(a.doc_id in {"s::3","s::4"} for a in actions if a.kind=="read")
    # Original v2 still works unchanged: regression isolation.
    legacy=build_legal_actions(q,obs,cfg=FiniteActionConfig(answer_proposer="v2"),
                               remaining=3,max_tool_calls=5)
    assert len(legacy)<=cfg.max_candidates


def test_v3_terminal_budget_exposes_only_factual_finish_and_abstain():
    obs=_observations()
    c=FiniteActionConfig(answer_proposer="v3")
    options=build_legal_actions("Who chaired the Vienna meeting?",obs,cfg=c,
                                remaining=1,max_tool_calls=5)
    assert len(options)>=2
    assert all(a.kind=="finish" for a in options)
    assert all(set(a.citations).issubset({"s::1","s::2"}) for a in options)
    assert len(options)<=c.max_candidates
    assert options[-1].answer==""


def test_train_curriculum_strictly_filters_without_dev_or_duplicates():
    qs=[Question(f"train{i}","question","answer",split="train")
        for i in range(20)]
    allowed={f"train{i}" for i in (1,3,5,7,11)}
    a=select_train_only(qs,set(),seed=43,n=4,allowed_train_ids=allowed)
    b=select_train_only(list(reversed(qs)),set(),seed=43,n=4,
                        allowed_train_ids=allowed)
    assert {q.question_id for q in a}.issubset(allowed)
    assert [q.question_id for q in a]==[q.question_id for q in b]
    with pytest.raises(ValueError,match="overlaps"):
        select_train_only(qs,{"train1"},seed=43,n=3,
                          allowed_train_ids=allowed)
    with pytest.raises(ValueError,match="belong"):
        select_train_only(qs,set(),seed=43,n=3,
                          allowed_train_ids={"not_a_train_question"})
    with pytest.raises(ValueError,match="not enough"):
        select_train_only(qs,set(),seed=43,n=6,allowed_train_ids=allowed)


def test_v3_rejects_unversioned_or_oracle_proposer():
    assert FiniteActionConfig(answer_proposer="v3").answer_proposer=="v3"
    with pytest.raises(ValueError,match="answer proposer"):
        FiniteActionConfig(answer_proposer="oracle")
