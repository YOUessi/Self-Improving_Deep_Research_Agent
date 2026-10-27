"""RSI campaigns are distinct frozen TRAIN queues, not learned generations."""
import hashlib
import json
from pathlib import Path

import pytest

from sidra.rsi_campaign_audit import audit_train_campaign
from sidra.rsi_challenge_scheduler import TrainingChallengeSeed,freeze_challenger_queue,_sha
from sidra.rsi_model_challenger import parse_challenge


TRAIN=[
    "Which scientist discovered the Aurora River source in the Arctic?",
    "In which decade did the Eastern Bridge get rebuilt after the storm?",
    "Which artist created the White Tower sculpture in Florence?",
    "Which city hosted the Amber Observatory inauguration ceremony?",
    "Where was the Meridian satellite research laboratory established?",
    "Who was the engineer responsible for the Crimson railway tunnel?",
]


def make_campaign(tmp_path,questions=TRAIN):
    paths=[]
    for i in range(2):
        examples=[]
        for j,q in enumerate(questions[2*i:2*i+2]):
            ix=2*i+j
            examples.append(TrainingChallengeSeed(
                opaque_challenge_id=f"c-{ix:024d}",
                source_train_id=f"train-{ix}",
                difficulty_tier=("bridge" if j else "hard"),
                normalized_text_hash=_sha(q),
                question=q))
        p=tmp_path/f"g{i+1}"
        freeze_challenger_queue(
            examples,p,source_git_sha="1"*40,seed=43+i,generation=i+1)
        paths.append(p)
    return paths


def make_candidates(tmp_path,queue_paths,*,cross_generation_duplicate=False):
    dirs=[]
    repeated=None
    for generation,queue in enumerate(queue_paths,1):
        rows=[json.loads(x) for x in
              (queue/"CHALLENGER_INPUT.jsonl").read_text().splitlines()]
        records=[]
        for idx,r in enumerate(rows):
            unique_questions=[
                "Which maritime archive reports the first launch of the Aurora polar research vessel?",
                "What independent geological survey established the age of the Eastern Bridge foundation?",
                "How did Renaissance patrons fund the restoration of the White Tower sculpture?",
                "Which original municipal decree authorized the Amber Observatory ceremony?",
            ]
            proposal=unique_questions[(generation-1)*2+idx]
            if cross_generation_duplicate and repeated is not None:
                proposal=repeated
            repeated=proposal if repeated is None else repeated
            reason="Need an independent archive and contemporaneous supporting documents."
            parsed=parse_challenge(
                json.dumps({"question":proposal,"reason":reason}),
                challenge_id=r["challenge_id"],source_question=r["question"])
            assert parsed.verdict=="unlabelled_candidate_pending_independent_verification"
            records.append(parsed.as_dict())
        p=tmp_path/f"candidates_{generation}"
        p.mkdir()
        (p/"UNLABELLED_CHALLENGER_CANDIDATES.jsonl").write_text(
            "\n".join(json.dumps(x) for x in records)+"\n")
        manifest=json.loads((queue/"QUEUE_MANIFEST.json").read_text())
        summary={
            "source_queue_generation":generation,
            "source_queue_sha256":manifest["challenger_input_sha256"],
            "source_split":"train","backend_local_only":True,
            "challenger_gold_label_access":False,
            "independent_verifier_approved_novel_tasks":0,
            "trainable_new_gold_labels_produced":0,"optimizer_updates":0,
            "n_attempted":len(records),
            "n_structurally_valid_novel_candidates":len(records),
            "verdict_histogram":{
                "unlabelled_candidate_pending_independent_verification":len(records)},
        }
        (p/"CHALLENGER_SUMMARY.json").write_text(json.dumps(summary))
        dirs.append(p)
    return dirs


def test_frozen_train_campaign_has_no_trainable_new_labels(tmp_path):
    dirs=make_campaign(tmp_path)
    result=audit_train_campaign(list(reversed(dirs)))
    assert result["queue_generations"]==[1,2]
    assert result["queue_seeds"]==[43,44]
    assert result["frozen_train_questions_total"]==4
    assert result["unique_train_sources_total"]==4
    assert result["difficulty_proxy_totals"]=={"hard":2,"bridge":2}
    assert result["new_trainable_tasks_granted"]==0
    assert not result["multi_generation_RSI_completed"]
    assert not result["model_origin_proven_by_this_audit"]
    assert "train-" not in json.dumps(result)  # no per-example ID leakage


def test_novel_model_candidates_stay_unlabelled(tmp_path):
    dirs=make_campaign(tmp_path)
    generated=make_candidates(tmp_path,dirs)
    summary=audit_train_campaign(dirs,candidate_dirs=list(reversed(generated)))
    assert summary["candidate_responses_checked"]==4
    assert summary["novel_structurally_valid_unlabelled_candidates"]==4
    assert summary["new_trainable_tasks_granted"]==0
    assert summary["multiple_optimizer_seed_replicates_verified"] is False


def test_corrupt_verifier_map_denied_even_when_queue_valid(tmp_path):
    dirs=make_campaign(tmp_path)
    f=dirs[1]/"VERIFIER_ONLY_MAP.jsonl"
    f.write_text(f.read_text().replace("train-2","dev-1"))
    with pytest.raises(ValueError,match="checksum"):
        audit_train_campaign(dirs)


def test_manifest_cannot_rebrand_nontrain_or_claim_training(tmp_path):
    dirs=make_campaign(tmp_path)
    path=dirs[0]/"QUEUE_MANIFEST.json"
    manifest=json.loads(path.read_text())
    manifest["training_or_checkpoint_updates"]=2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="TRAIN"):
        audit_train_campaign(dirs)


def test_identity_leakage_requires_exact_isolated_queue_shape(tmp_path):
    dirs=make_campaign(tmp_path)
    f=dirs[0]/"CHALLENGER_INPUT.jsonl"
    rows=[json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["gold_answer"]="secret"
    f.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    manifest_file=dirs[0]/"QUEUE_MANIFEST.json"
    manifest=json.loads(manifest_file.read_text())
    manifest["challenger_input_sha256"]=hashlib.sha256(f.read_bytes()).hexdigest()
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="unexpected fields"):
        audit_train_campaign(dirs)


def test_cross_generation_semantic_near_duplicate_guard(tmp_path):
    q=list(TRAIN)
    q[2]=q[0].replace("Aurora River source","Aurora River headwaters")
    dirs=make_campaign(tmp_path,q)
    with pytest.raises(ValueError,match="near-duplicate"):
        audit_train_campaign(dirs,near_duplicate_threshold=0.83)


def test_repeated_model_proposal_in_future_generation_denied(tmp_path):
    dirs=make_campaign(tmp_path)
    generated=make_candidates(tmp_path,dirs,cross_generation_duplicate=False)
    f=generated[1]/"UNLABELLED_CHALLENGER_CANDIDATES.jsonl"
    rows=[json.loads(x) for x in f.read_text().splitlines()]
    original=json.loads((generated[0]/"UNLABELLED_CHALLENGER_CANDIDATES.jsonl")
                        .read_text().splitlines()[0])
    rows[0]["proposal_question"]=original["proposal_question"]
    f.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    with pytest.raises(ValueError,match="independently rejected"):
        audit_train_campaign(dirs,candidate_dirs=generated)


def test_fake_candidate_training_label_or_summary_denied(tmp_path):
    dirs=make_campaign(tmp_path)
    generated=make_candidates(tmp_path,dirs)
    mf=generated[1]/"CHALLENGER_SUMMARY.json"
    data=json.loads(mf.read_text())
    data["trainable_new_gold_labels_produced"]=1
    mf.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="TRAIN isolation"):
        audit_train_campaign(dirs,candidate_dirs=generated)


def test_campaign_generations_must_be_contiguous_and_seeds_distinct(tmp_path):
    dirs=make_campaign(tmp_path)
    path=dirs[1]/"QUEUE_MANIFEST.json"
    manifest=json.loads(path.read_text())
    manifest["generation"]=3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="contiguous"):
        audit_train_campaign(dirs)
