"""E-029 frozen TRAIN annotation packets, privacy guard and declared review parsing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sidra import blind_adjudication as review
from sidra.schema import Question


def fixture(tmp_path,monkeypatch):
    # 12 used in E-026 + 8 new disjoint original TRAIN examples.
    qs=[];original=[]
    for i in range(20):
        qid=f"TRAIN-id-{i:03d}"
        kind="bridge" if i%2==0 else "comparison"
        question=f"In original source {i}, which team won the match?"
        answer=f"Team-{i}"
        qs.append(Question(question_id=qid,question=question,
                           gold_answer=answer,split="train"))
        original.append({
            "_id":qid,"question":question,"answer":answer,"type":kind,
            "supporting_facts":[["Wiki One",0],["Wiki Two",0]],
            "context":[
                ["Wiki One",[f"The match took place in venue {i}."]],
                ["Wiki Two",[f"Team-{i} won the match."]],
            ],
        })
    from sidra import schema
    monkeypatch.setattr(review,"load_dataset",lambda dataset:([],qs))
    data=tmp_path/"data";data.mkdir()
    for name in ("questions.jsonl","documents.jsonl"):
        (data/name).write_text("{}\n")
    original_file=tmp_path/"train_original.json"
    original_file.write_text(json.dumps(original))
    e026=tmp_path/"e026.jsonl"
    e026.write_text("\n".join(json.dumps({
        "case_id":f"case-{i}-{j}","question_id":qs[i].question_id,
    }) for i in range(12) for j in range(3))+"\n")
    return data,original_file,e026


def make_packet(tmp_path,monkeypatch):
    data,original,e026=fixture(tmp_path,monkeypatch)
    output=tmp_path/"packet"
    meta=review.freeze_blind_packet(
        dataset=data,original_train=original,
        prior_e026_rows=e026,out=output,seed=2029,
        bridge_count=2,comparison_count=2)
    return output,meta


def test_freeze_private_registry_and_two_equivalent_blind_packets(tmp_path,monkeypatch):
    out,meta=make_packet(tmp_path,monkeypatch)
    a=[json.loads(x) for x in (out/"REVIEWER_A_BLIND.jsonl").read_text().splitlines()]
    b=[json.loads(x) for x in (out/"REVIEWER_B_BLIND.jsonl").read_text().splitlines()]
    registry=[json.loads(x) for x in
              (out/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl").read_text().splitlines()]
    assert meta["question_count"]==4
    assert meta["variant_balance_private"]=={
        "full_sources":2,"one_support_title_omitted":2}
    assert {x["claim_id"] for x in a}=={x["claim_id"] for x in b}
    assert {x["claim_id"] for x in a}=={x["claim_id"] for x in registry}
    assert all(len(x["provided_evidence"]) in (1,2) for x in a)
    assert all("expected_label" not in x and "variant" not in x and
               "question_id" not in x and "model_posterior" not in x and
               "gold_answer" not in x for x in a+b)
    assert meta["independent_human_adjudication_completed"] is False
    assert not meta["allow_as_verifier_threshold_calibration"]
    assert all(int(x["source_train_question_id"].split("-")[-1])>=12 for x in registry)
    assert len({x["source_train_question_id"] for x in registry})==4
    assert meta["reviewer_packet_sha256"]["A"]==review.file_sha256(
        out/"REVIEWER_A_BLIND.jsonl")


def test_packet_overwrite_and_nontrain_refusal(tmp_path,monkeypatch):
    out,meta=make_packet(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError):
        review.freeze_blind_packet(
            dataset=tmp_path/"data",original_train=tmp_path/"train_original.json",
            prior_e026_rows=tmp_path/"e026.jsonl",out=out,
            bridge_count=2,comparison_count=2)
    assert meta["human_labels_received"]==0


def test_review_interpretation_requires_distinct_rater_and_locked_ids(tmp_path,monkeypatch):
    out,meta=make_packet(tmp_path,monkeypatch)
    cards=[json.loads(x) for x in (out/"REVIEWER_A_BLIND.jsonl").read_text().splitlines()]
    a=tmp_path/"r_a.jsonl";b=tmp_path/"r_b.jsonl"
    def write(path,rid,mutate_id=False,rating="supported"):
        lines=[{"claim_id":x["claim_id"],
                "rating":rating if i else "insufficient_evidence",
                "reviewer_id":rid,"human_attestation":True}
               for i,x in enumerate(cards)]
        if mutate_id:lines[0]["claim_id"]="forged-id"
        path.write_text("\n".join(json.dumps(x) for x in lines)+"\n")
    write(a,"human-rater-A");write(b,"human-rater-B")
    r=review.compare_independent_reviews(
        review_a=a,review_b=b,manifest=out/"MANIFEST.json")
    assert r["n"]==4
    assert r["raw_agreement_rate"]==1
    assert r["all_cases_gold_ready"] is False
    assert r["independently_human_annotated_confirmed_by_software"] is False
    write(b,"human-rater-A")
    with pytest.raises(ValueError,match="independent declared"):
        review.compare_independent_reviews(
            review_a=a,review_b=b,manifest=out/"MANIFEST.json")
    write(b,"human-rater-B",mutate_id=True)
    with pytest.raises(ValueError,match="same cases|locked blind"):
        review.compare_independent_reviews(
            review_a=a,review_b=b,manifest=out/"MANIFEST.json")
    write(b,"human-rater-B",rating="contradicted")
    r=review.compare_independent_reviews(
        review_a=a,review_b=b,manifest=out/"MANIFEST.json")
    assert len(r["disagreements_require_third_party_resolution"])==3
    assert r["all_cases_gold_ready"] is False


def test_known_sft_and_online_train_exclusion_is_proven(tmp_path,monkeypatch):
    data,original,e026=fixture(tmp_path,monkeypatch)
    sft=tmp_path/"previous_sft_actions.jsonl"
    sft.write_text(json.dumps({"question_id":"TRAIN-id-012"})+"\n")
    grpo=tmp_path/"prior_online_train.jsonl"
    grpo.write_text(json.dumps({"qid":"TRAIN-id-013"})+"\n")
    out=tmp_path/"packet_excluding_train"
    meta=review.freeze_blind_packet(
        dataset=data,original_train=original,prior_e026_rows=e026,
        out=out,seed=2029,bridge_count=2,comparison_count=2,
        prior_train_action_files=(sft,grpo))
    selected={json.loads(x)["source_train_question_id"] for x in
              (out/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl").read_text().splitlines()}
    assert not (selected & {"TRAIN-id-012","TRAIN-id-013"})
    assert meta["prior_known_project_train_question_excluded"]==2
    assert meta["disjoint_from_supplied_prior_train_artifacts_verified"] is True
    assert meta["project_train_exclusion_paths_count"]==2
    assert meta["prior_training_artifact_sha256"][str(sft)]==review.file_sha256(sft)


def test_prior_rollout_cannot_include_test_or_unknown_question(tmp_path,monkeypatch):
    data,original,e026=fixture(tmp_path,monkeypatch)
    prior=tmp_path/"invalid_prior.jsonl"
    prior.write_text(json.dumps({"question_id":"dev-outside"})+"\n")
    with pytest.raises(ValueError,match="non-TRAIN"):
        review.freeze_blind_packet(
            dataset=data,original_train=original,prior_e026_rows=e026,
            out=tmp_path/"not-created",bridge_count=2,comparison_count=2,
            prior_train_action_files=(prior,))
    assert not (tmp_path/"not-created").exists()
