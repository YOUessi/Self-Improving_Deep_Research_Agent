"""TRAIN-only observed-state reward is computed after real source actions."""
import pytest

from sidra.evidence_state_credit import assign_evidence_state_credit
from sidra.schema import Question, Document, Action, AgentTrace
from sidra.train_reward import score_train_rollout


Q=Question("train-q","Which province contains River City?",
           "Coastal Province",supporting_doc_ids=("a","b"),
           split="train")
DOCS={
    "a":Document("a","River City","River City is in Coastal Province."),
    "b":Document("b","Coastal Province","Coastal Province contains River City."),
    "d":Document("d","Unrelated Castle","An unrelated monument.")
}


def episode(rewarding:bool)->dict:
    found=["a","b"] if rewarding else ["d","a"]
    choices=[
        Action("search",query=Q.question),
        Action("read",doc_id=found[0]),
        Action("read",doc_id=found[1]),
        Action("finish",answer=Q.gold_answer,citations=("a","b"))
        if rewarding else Action("finish",answer="",citations=()),
    ]
    observations=[
        {"kind":"search","query":Q.question,
         "hits":[{"doc_id":d,"title":DOCS[d].title,
                  "snippet":DOCS[d].text[:100]} for d in found]},
        {"kind":"read","doc_id":found[0],"title":DOCS[found[0]].title,
         "text":DOCS[found[0]].text},
        {"kind":"read","doc_id":found[1],"title":DOCS[found[1]].title,
         "text":DOCS[found[1]].text},
    ]
    steps=[]
    samples=[]
    for i,a in enumerate(choices):
        entry={"step":i,"action":a.to_dict()}
        if i<len(observations):
            entry["observation"]=observations[i]
        steps.append(entry)
        samples.append({
            "legal_actions":[a.to_dict()],
            "selected_index":0,"selection_mode":"sample",
            "action_token_targets":1,"tool_observation_targets":0,
            "sampled_logprob":0.0,
        })
    trace=AgentTrace(
        question_id=Q.question_id,question=Q.question,
        policy="finite",steps=steps,
        answer=Q.gold_answer if rewarding else "",
        citations=["a","b"] if rewarding else [],
        termination="finish" if rewarding else "abstained",
        search_calls=1,read_calls=2,tool_calls=3)
    reward=score_train_rollout(Q,trace,DOCS)
    return {"qid":Q.question_id,"trace":trace.as_dict(),
            "train_reward":reward.as_dict(),"action_samples":samples}


def test_reward_uses_same_state_as_policy_and_is_question_normalized():
    episodes=[episode(True),episode(False),episode(True),episode(False)]
    audit=assign_evidence_state_credit(episodes,Q)
    assert audit["mode"]=="evidence_state_delta_v1"
    assert audit["policy_prompt_or_catalog_sees_original_labels"] is False
    assert audit["tool_observation_label_tokens"]==0
    assert audit["nonzero_optimizer_action_advantages"]>0
    assert audit["by_kind"]["read"]["nonzero_state_delta"]>=4
    for ep in episodes:
        assert len(ep["action_samples"])==4
        for sample in ep["action_samples"]:
            assert "evidence_state_before" in sample
            assert "evidence_state_after" in sample
            assert "step_credit_advantage" in sample
            assert "gold_answer" not in str(sample["evidence_state_before"])
            assert "supporting_doc_ids" not in str(sample["evidence_state_after"])
            assert sample["tool_observation_targets"]==0
    success=episodes[0]["action_samples"][-1]
    failed=episodes[1]["action_samples"][-1]
    assert success["step_credit_advantage"]>failed["step_credit_advantage"]
    assert success["evidence_state_delta_raw"]==0


def test_reward_refuses_dev_gold_and_nonfinite_mixing():
    with pytest.raises(ValueError,match="TRAIN"):
        assign_evidence_state_credit(
            [episode(True),episode(False)],
            Question("dev","Q","A",split="dev",supporting_doc_ids=("a","b")))
    with pytest.raises(ValueError,match="weight"):
        assign_evidence_state_credit([episode(True),episode(False)],Q,
                                     state_delta_weight=1.1)


def test_false_action_trace_never_receives_train_credit():
    episodes=[episode(True),episode(False)]
    episodes[0]["action_samples"][0]["legal_actions"][0]["query"]="FAKE QUERY"
    with pytest.raises(ValueError,match="mismatch"):
        assign_evidence_state_credit(episodes,Q)


def test_v2_reward_refuses_to_count_unrelated_distractor_read():
    episodes=[episode(True),episode(False),episode(True),episode(False)]
    audit=assign_evidence_state_credit(
        episodes,Q,state_version="v2",state_delta_weight=0.25)
    assert audit["state_version"]=="v2"
    assert audit["mode"]=="evidence_state_delta_v2"
    assert audit["nonzero_optimizer_action_advantages"]>0
    # Invalid/missing source labels still cannot be passed into the policy
    # prompt, only the after-episode TRAIN audit.
    for ep in episodes:
        for action in ep["action_samples"]:
            assert action["evidence_state_before"]["state_version"]=="v2"
            assert action["tool_observation_targets"]==0
    with pytest.raises(ValueError,match="version"):
        assign_evidence_state_credit(episodes,Q,state_version="v0")


def test_v3_reward_suppresses_observable_but_non_gold_anchor_progress():
    import copy
    episodes=[episode(True),episode(False),episode(True),episode(False)]
    # A second document bearing the SAME question-anchor title is really
    # READ, but it is NOT one of the original TRAIN support IDs.
    fake=episodes[1]
    first_hit=fake["trace"]["steps"][0]["observation"]["hits"][0]
    assert first_hit["doc_id"]=="d"
    first_hit.update({"doc_id":"x","title":"River City",
                      "snippet":"River City was founded in 1700."})
    fake_step=fake["trace"]["steps"][1]
    fake_step["action"]["doc_id"]="x"
    fake_step["observation"].update({
        "doc_id":"x","title":"River City",
        "text":"River City was founded in 1700. No other relation here.",
    })
    fake["action_samples"][1]["legal_actions"][0]["doc_id"]="x"
    audit=assign_evidence_state_credit(episodes,Q,state_version="v3")
    assert audit["mode"]=="evidence_state_delta_v3"
    assert audit["policy_prompt_or_catalog_sees_original_labels"] is False
    wrong_read=fake["action_samples"][1]
    assert wrong_read["evidence_state_delta_raw"]>0
    assert wrong_read["evidence_state_delta_credited_raw"]==0
    assert wrong_read["step_state_train_reward_raw"]==0
    assert audit["by_kind"]["read"]["suppressed_state_delta_total"]>0
    true_read=episodes[0]["action_samples"][1]
    assert true_read["step_state_train_reward_raw"]>0
    assert episodes[0]["action_samples"][-1]["step_state_train_reward_raw"]==1
    assert episodes[1]["action_samples"][-1]["step_state_train_reward_raw"]==0
    assert all(x["tool_observation_targets"]==0
               for ep in episodes for x in ep["action_samples"])


def test_v3_training_credit_must_match_v3_policy_version():
    episodes=[episode(True),episode(False)]
    assert assign_evidence_state_credit(episodes,Q,state_version="v3")[
        "state_version"]=="v3"
    with pytest.raises(ValueError,match="TRAIN"):
        assign_evidence_state_credit(episodes,
            Question("dev","Q","A",split="dev",supporting_doc_ids=("a","b")),
            state_version="v3")
