"""Real Challenger requires exact independent TRAIN text quotes; no auto gold."""
import json

import pytest

from sidra.retrieval import BM25Index
from sidra.rsi_challenge_scheduler import TrainingChallengeSeed,freeze_challenger_queue,_sha
from sidra.rsi_grounded_challenger import (
    EvidenceExcerpt,GroundingPacket,make_grounding_packet,
    grounded_messages,verify_grounded_response,
)
from sidra.schema import Document,Question
from scripts.run_rsi_grounded_challenger import execute


EXCERPTS=(
    EvidenceExcerpt("d1","Mason Bridge",
        "The Mason Bridge was designed by architect Elena Hart in 1982."),
    EvidenceExcerpt("d2","Elena Hart awards",
        "Elena Hart received the Civic Engineering Medal in September 1991."),
)


def pkt()->GroundingPacket:
    return GroundingPacket("c-001","Which bridge did Elena Hart design?",
                           EXCERPTS,"a"*64)


def candidate():
    return {
        "question":"Which civic medal was given to the designer of Mason Bridge in 1991?",
        "proposed_answer":"Civic Engineering Medal",
        "reason":"Must connect the bridge's designer with her separately listed award.",
        "evidence":[
            {"doc_id":"d1","quote":EXCERPTS[0].snippet},
            {"doc_id":"d2","quote":EXCERPTS[1].snippet},
        ],
    }


def test_exact_cross_document_quotes_are_only_pending_semantic_review():
    raw=json.dumps(candidate())
    x=verify_grounded_response(raw,pkt())
    assert x.verdict=="quote_extracted_unlabelled_pending_semantic_verification"
    assert x.proposed_answer=="Civic Engineering Medal"
    assert {d["doc_id"] for d in x.evidence}=={"d1","d2"}
    assert x.completion_sha256 and x.packet_sha256=="a"*64
    assert "gold_answer" not in json.dumps(x.as_dict())


def test_quote_forgery_and_duplicate_docs_denied():
    p=candidate()
    p["evidence"][0]["quote"]="The Mason Bridge was designed by the wrong person in 1982."
    assert verify_grounded_response(
        json.dumps(p),pkt()).verdict=="quote_not_exact_tool_observation"
    p=candidate()
    p["evidence"][1]["doc_id"]="d1"
    p["evidence"][1]["quote"]=EXCERPTS[0].snippet
    assert verify_grounded_response(
        json.dumps(p),pkt()).verdict=="unretrieved_or_duplicate_citation"


def test_answer_must_literally_appear_in_quote():
    p=candidate()
    p["proposed_answer"]="Never Actually Awarded Prize"
    assert verify_grounded_response(
        json.dumps(p),pkt()).verdict=="answer_not_extractive_in_provided_quotes"


def test_model_abstention_and_hidden_gold_fields_never_training():
    p=candidate()
    p["evidence"]=[]
    assert verify_grounded_response(json.dumps(p),pkt()).verdict=="model_abstained"
    p=candidate()
    p["gold_answer"]="Civic Engineering Medal"
    assert verify_grounded_response(json.dumps(p),pkt()).verdict=="invalid_schema"


def test_grounding_packet_only_contains_question_and_retrieved_text():
    docs=[
        Document("d1",EXCERPTS[0].title,EXCERPTS[0].snippet,scope="train-1"),
        Document("d2",EXCERPTS[1].title,EXCERPTS[1].snippet,scope="train-1"),
        Document("private_doc","Forbidden","private token",scope="dev-1"),
    ]
    idx=BM25Index(docs)
    packet=make_grounding_packet(
        "c-001","Which bridge did Elena Hart design?",
        scope="train-1",index=idx,top_k=4)
    view=packet.model_view()
    data=json.dumps(view,ensure_ascii=False)
    assert "gold_answer" not in data
    assert "supporting_doc_ids" not in data
    assert "private token" not in data
    assert "private_doc" not in data
    assert len(packet.snippets)==2
    assert len(packet.packet_sha256)==64
    assert len(grounded_messages(packet))==2


def test_invalid_model_endpoint_inherits_local_only_trust_gate():
    from sidra.rsi_grounded_challenger import LocalGroundedChallenger
    with pytest.raises(ValueError,match="local"):
        LocalGroundedChallenger("https://example.com/v1/chat/completions","8b")


def test_grounded_cli_keeps_generated_answers_out_of_training(tmp_path):
    docs=[
        Document("d1",EXCERPTS[0].title,EXCERPTS[0].snippet,scope="train-1"),
        Document("d2",EXCERPTS[1].title,EXCERPTS[1].snippet,scope="train-1"),
    ]
    question="Which bridge did Elena Hart design?"
    q=Question("train-1",question,"Mason Bridge",("d1","d2"),
               split="train",scope="train-1")
    data=tmp_path/"train"
    data.mkdir()
    (data/"documents.jsonl").write_text(
        "\n".join(json.dumps(d.__dict__) for d in docs)+"\n")
    (data/"questions.jsonl").write_text(json.dumps(q.__dict__)+"\n")
    seed=TrainingChallengeSeed(
        "c-001","train-1","direct",_sha(question),question)
    queue=tmp_path/"queue"
    freeze_challenger_queue(
        [seed],queue,source_git_sha="a"*40,seed=43,generation=1)
    class FakeLocalBackend:
        model="qwen-local-mock"
        def generate_packet(self,packet,*,seed):
            assert seed==43
            view=json.dumps(packet.model_view())
            assert "gold_answer" not in view
            assert "supporting_doc_ids" not in view
            assert len(packet.snippets)==2
            return json.dumps(candidate())
    out=tmp_path/"results"
    report=execute(
        train_data=data,queue_file=queue/"CHALLENGER_INPUT.jsonl",
        manifest_file=queue/"QUEUE_MANIFEST.json",
        verifier_map=queue/"VERIFIER_ONLY_MAP.jsonl",
        backend=FakeLocalBackend(),seed=43,out=out,limit=1)
    assert report["observed_quote_structural_passes"]==1
    assert report["new_trainable_challenges_approved"]==0
    assert report["semantic_entailment_verified"] is False
    assert report["gold_answer_or_supporting_ids_in_model_prompts"] is False
    with pytest.raises(FileExistsError):
        execute(train_data=data,queue_file=queue/"CHALLENGER_INPUT.jsonl",
                manifest_file=queue/"QUEUE_MANIFEST.json",
                verifier_map=queue/"VERIFIER_ONLY_MAP.jsonl",
                backend=FakeLocalBackend(),seed=43,out=out,limit=1)

    # Independent verifier runs AFTER inference, from saved immutable
    # artefacts and original TRAIN text. Model self-reported "valid" does
    # not confer any gold-label authority.
    from sidra.rsi_grounded_audit import audit_saved_grounded_candidates
    from hashlib import sha256

    def check():
        return audit_saved_grounded_candidates(
            data,queue/"CHALLENGER_INPUT.jsonl",
            queue/"QUEUE_MANIFEST.json",
            queue/"VERIFIER_ONLY_MAP.jsonl",out)
    checked=check()
    assert checked["literal_two_document_quote_passes"]==1
    assert checked["new_trainable_challenges_approved"]==0
    assert checked["semantic_entailment_approved"]==0
    assert checked["grounding_packets_recomputed"]==1

    result_file=out/"UNLABELLED_GROUNDED_CANDIDATES.jsonl"
    summary_file=out/"GROUNDED_CANDIDATE_SUMMARY.json"
    original_rows=result_file.read_text()
    original_summary=summary_file.read_text()
    rows=[json.loads(x) for x in original_rows.splitlines()]
    rows[0]["evidence"][0]["quote"]="The bridge was built by the incorrect designer."
    result_file.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    metrics=json.loads(summary_file.read_text())
    metrics["grounded_candidates_output_sha256"]=sha256(result_file.read_bytes()).hexdigest()
    summary_file.write_text(json.dumps(metrics))
    with pytest.raises(ValueError,match="quote proof"):
        check()
    result_file.write_text(original_rows)
    summary_file.write_text(original_summary)

    # An attacker must not mark a quote-valid novel question as fully
    # verified merely by editing its aggregate manifest.
    metrics=json.loads(summary_file.read_text())
    metrics["new_trainable_challenges_approved"]=1
    summary_file.write_text(json.dumps(metrics))
    with pytest.raises(ValueError,match="unsafe model label claim"):
        check()
