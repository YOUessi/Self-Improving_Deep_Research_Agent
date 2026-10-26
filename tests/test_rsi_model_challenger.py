"""No privileged labels enter model-driven Challenger; proposals stay UNLABELLED."""
import hashlib,json

import pytest

from sidra.rsi_model_challenger import (
    LocalOpenAIChallenger,challenger_messages,generate_train_candidates,
    parse_challenge,
)


def test_model_challenger_prompt_never_contains_evaluator_labels():
    messages=challenger_messages("What is the city of Mason Bridge?")
    assert len(messages)==2
    full=json.dumps(messages)
    assert "gold_answer" not in full
    assert "supporting_doc_ids" not in full
    assert "DEV" not in messages[1]["content"]
    assert "final-B160" not in messages[1]["content"]


@pytest.mark.parametrize("endpoint",[
    "https://evil.example/v1/chat/completions",
    "http://evil.example/v1/chat/completions",
    "http://127.0.0.1:18985/anything_else",
    "file:///etc/passwd",
    "http://user:pass@127.0.0.1:18985/v1/chat/completions",
])
def test_refuse_nonlocal_endpoints(endpoint):
    with pytest.raises(ValueError,match="local"):
        LocalOpenAIChallenger(endpoint,"model")


def test_structural_acceptance_without_ground_truth_credit():
    original="What city has the Mason Bridge spanning its river?"
    proposal={"question":"Which bridge engineer influenced later designs in the Mason Bridge region?",
              "reason":"Need to find engineer and historically supported design influence."}
    outcome=parse_challenge(json.dumps(proposal),
                            challenge_id="c-1",source_question=original)
    assert outcome.verdict=="unlabelled_candidate_pending_independent_verification"
    assert outcome.proposal_question==proposal["question"]
    assert outcome.model_completion_sha256


def test_repeat_and_hidden_answer_contamination_rejected():
    original="What city is home to the Mason Bridge?"
    o=parse_challenge(json.dumps({"question":original,"reason":"A long research task"}),
                      challenge_id="c",source_question=original)
    assert o.verdict=="repeats_original_question"
    bad={"question":"What city contains the hidden gold_answer for Mason Bridge?",
         "reason":"We can read the label to generate task"}
    o=parse_challenge(json.dumps(bad),challenge_id="c",source_question=original)
    assert o.verdict=="references_hidden_labels"
    assert o.proposal_question==""


def test_malformed_challenger_actions_never_approved():
    out=parse_challenge('{"kind":"finish","answer":"secret"}',
                        challenge_id="c",source_question="What place holds Mason Bridge?")
    assert out.verdict=="invalid_schema"
    out=parse_challenge("Just a sentence",challenge_id="c",
                        source_question="What place holds Mason Bridge?")
    assert out.verdict=="malformed_json"


def test_duplicate_previous_model_proposals_rejected():
    proposal="Which architect designed the bridge outside Oxfordshire Castle?"
    encoded=json.dumps({"question":proposal,
                        "reason":"Require independent research for bridge architect."})
    x=parse_challenge(encoded,challenge_id="id",
                      source_question="How long was the Oxfordshire Castle wall?",
                      prior_proposals=[proposal])
    assert x.verdict=="near_duplicate_previous_candidate"


def test_real_model_generator_is_isolated_by_input_contract(tmp_path):
    questions=[
        {"challenge_id":"a","question":"What city contains Mason Bridge and why is it named so?"},
        {"challenge_id":"b","question":"Who funded the construction of Marina Tower?"},
    ]
    queue=tmp_path/"CHALLENGER_INPUT.jsonl"
    queue.write_text("\n".join(json.dumps(q) for q in questions)+"\n")
    fp=hashlib.sha256(queue.read_bytes()).hexdigest()
    manifest=tmp_path/"QUEUE_MANIFEST.json"
    manifest.write_text(json.dumps({
        "challenger_input_sha256":fp,"split":"train","n":2,
        "generation":1,"model_driven_challenger_executed":False}))
    class StubBackend:
        model="test-model"
        seen=[]
        def generate(self,source_question,*,seed):
            self.seen.append((source_question,seed))
            return json.dumps({
                "question":"Which historical document verified the construction date of "+(
                    "Mason Bridge" if "Bridge" in source_question else "Marina Tower")+"?",
                "reason":"Need original historical sources and independent corroboration.",
            })
    backend=StubBackend()
    summary=generate_train_candidates(
        queue,manifest,tmp_path/"result",backend=backend,seed=43)
    assert summary["n_attempted"]==2
    assert summary["n_structurally_valid_novel_candidates"]==2
    assert summary["trainable_new_gold_labels_produced"]==0
    assert summary["optimizer_updates"]==0
    assert backend.seen==[(x["question"],43+i) for i,x in enumerate(questions)]
    for line in (tmp_path/"result/UNLABELLED_CHALLENGER_CANDIDATES.jsonl").read_text().splitlines():
        row=json.loads(line)
        assert "gold_answer" not in json.dumps(row)
        assert "supporting_doc_ids" not in json.dumps(row)
    with pytest.raises(FileExistsError):
        generate_train_candidates(queue,manifest,tmp_path/"result",
                                  backend=backend,seed=44)


def test_challenger_rejects_extra_input_fields(tmp_path):
    queue=tmp_path/"q.jsonl"
    queue.write_text(json.dumps({
        "challenge_id":"q","question":"Who built the ancient bridge?",
        "gold_answer":"Someone"})+"\n")
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({
        "split":"train","generation":1,"n":1,
        "model_driven_challenger_executed":False,
        "challenger_input_sha256":hashlib.sha256(queue.read_bytes()).hexdigest()}))
    class NeverCalled:
        model="dummy"
        def generate(self,*args,**kwargs):
            raise AssertionError("no model calls may happen on leaking input")
    with pytest.raises(ValueError,match="leaked"):
        generate_train_candidates(
            queue,manifest,tmp_path/"out",backend=NeverCalled(),seed=1)
