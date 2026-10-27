"""E-027 regression: sanitized Solver never reads TRAIN gold; offline verifier does."""
import hashlib
import json
from pathlib import Path

import pytest

from sidra.schema import Action
from sidra.rsi_solver_cohort import (
    _validate_public_pack,doc_alias,prepare_solver_pack,
    run_real_solver,audit_solver_run,
)


SHA="a"*40
CID="c-"+"1"*24
QID="train-001"


def rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in values),
                    encoding="utf-8")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case(tmp_path):
    train=tmp_path/"canonical_train"
    train.mkdir()
    docs=[
        {"doc_id":QID+"::0","title":"Harbor River","text":"Harbor River flows into Harbor.","scope":QID},
        {"doc_id":QID+"::1","title":"Harbor Town","text":"Harbor is a city, and Harbor River enters the bay.","scope":QID},
        {"doc_id":"unseen-id::0","title":"Hidden Question","text":"This document must never appear.","scope":"unseen-id"},
    ]
    questions=[{
        "question_id":QID,"question":"Where does Harbor River flow?",
        "gold_answer":"Harbor","supporting_doc_ids":[QID+"::0",QID+"::1"],
        "scope":QID,"group":"train","split":"train",
    }]
    rows(train/"documents.jsonl",docs)
    rows(train/"questions.jsonl",questions)
    campaign=tmp_path/"campaign"
    sub=campaign/"cohort_01"
    prompts=[{"challenge_id":CID,"question":"Where does Harbor River flow?"}]
    mapping=[{"challenge_id":CID,"source_train_question_id":QID,
              "text_sha256":hashlib.sha256(
                  prompts[0]["question"].casefold().encode()).hexdigest(),
              "tier":"direct"}]
    rows(sub/"SOLVER_INPUT.jsonl",prompts)
    rows(sub/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl",mapping)
    manifest={
        "source_git_sha":SHA,
        "original_train_documents_sha256":checksum(train/"documents.jsonl"),
        "cohorts":[{
            "planned_curriculum_cohort":1,
            "solver_input_sha256":checksum(sub/"SOLVER_INPUT.jsonl"),
            "verifier_only_map_sha256":checksum(sub/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl"),
            "n_original_train_questions":1
        }]
    }
    (campaign/"CAMPAIGN_MANIFEST.json").write_text(json.dumps(manifest))
    return train,campaign


class FakeObservedLLM:
    """NOT A MODEL; used only by CPU integration fixtures."""
    name="fake_test_policy"
    def __init__(self,*args,**kwargs):
        self.model_calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.action_io=[]
    def decide(self,question,observations,remaining):
        assert question=="Where does Harbor River flow?"
        self.model_calls+=1
        self.action_io.append({"messages":[{"role":"user","content":question}],
                               "raw_completion":"CPU TEST FIXTURE ONLY"})
        self.prompt_tokens+=20
        self.completion_tokens+=10
        found=[h["doc_id"] for o in observations if o["kind"]=="search"
               for h in o["hits"]]
        read=[o["doc_id"] for o in observations if o["kind"]=="read"]
        if not found:return Action("search",query="Harbor River Harbor")
        if len(read)<2:
            return Action("read",doc_id=next(x for x in found if x not in read))
        return Action("finish",answer="Harbor",citations=tuple(read))


def test_training_labels_never_enter_model_pack_and_real_tool_replay_is_independent(tmp_path,monkeypatch):
    train,campaign=case(tmp_path)
    pack=tmp_path/"model_only"
    pm=prepare_solver_pack(train/"documents.jsonl",campaign,1,pack,source_sha=SHA)
    assert pm["questions"]==1 and pm["documents"]==2
    assert not (pack/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl").exists()
    for filename in ("SOLVER_INPUT.jsonl","SOLVER_DOCUMENTS.jsonl"):
        data=(pack/filename).read_text()
        assert "gold_answer" not in data
        assert "supporting_doc_ids" not in data
        assert QID not in data
        assert "Hidden Question" not in data
    _,prompt,docs=_validate_public_pack(pack)
    assert prompt==[{"challenge_id":CID,"question":"Where does Harbor River flow?"}]
    assert all(d.scope==CID for d in docs)
    from sidra import policies
    monkeypatch.setattr(policies,"OpenAICompatiblePolicy",FakeObservedLLM)
    output=tmp_path/"real_model_fixture"
    # Synthetic policy only; test validates the true SearchAgent / tool API path.
    model_report=run_real_solver(
        pack,output,endpoint="http://127.0.0.1:19999/v1/chat/completions",
        model="local-only-test",checkpoint_sha256="b"*64,
        source_sha=SHA)
    assert model_report["n_episodes"]==1
    assert model_report["optimizer_updates"]==0
    assert model_report["TRAIN_gold_loaded_or_graded_by_solver"] is False
    audit=audit_solver_run(train,campaign,pack,output,source_sha=SHA)
    assert audit["n"]==1 and audit["tool_replay_valid"]==1
    assert audit["legal_finished_answers"]==1
    assert audit["strict_original_TRAIN_successes"]==1
    assert audit["multi_generation_RSI_proven"] is False
    assert "gold_answer" not in (output/"MODEL_ACTION_IO.jsonl").read_text()
    assert "supporting_doc_ids" not in (output/"MODEL_ACTION_IO.jsonl").read_text()


def test_corrupt_model_pack_rejected_before_inference(tmp_path):
    train,campaign=case(tmp_path)
    pack=tmp_path/"model_only"
    prepare_solver_pack(train/"documents.jsonl",campaign,1,pack,source_sha=SHA)
    p=pack/"SOLVER_INPUT.jsonl"
    orig=json.loads(p.read_text().splitlines()[0])
    rows(p,[{**orig,"gold_answer":"leak"}])
    with pytest.raises(ValueError,match="changed after freezing"):
        _validate_public_pack(pack)
    with pytest.raises(FileExistsError):
        prepare_solver_pack(train/"documents.jsonl",campaign,1,pack,source_sha=SHA)


def test_hash_laundered_corpus_tamper_is_refused_by_independent_verifier(tmp_path,monkeypatch):
    train,campaign=case(tmp_path)
    pack=tmp_path/"model_only"
    prepare_solver_pack(train/"documents.jsonl",campaign,1,pack,source_sha=SHA)
    from sidra import policies
    monkeypatch.setattr(policies,"OpenAICompatiblePolicy",FakeObservedLLM)
    output=tmp_path/"real_fixture"
    run_real_solver(pack,output,endpoint="http://127.0.0.1:19999/v1/chat/completions",
                    model="stub",checkpoint_sha256="b"*64,source_sha=SHA)
    d=pack/"SOLVER_DOCUMENTS.jsonl"
    items=[json.loads(x) for x in d.read_text().splitlines()]
    items[0]["text"]="Forged evidence which was never in TRAIN."
    rows(d,items)
    m=pack/"SOLVER_PACK_MANIFEST.json"
    man=json.loads(m.read_text())
    man["documents_sha256"]=checksum(d)
    m.write_text(json.dumps(man))
    # Re-sign solver model-run pack reference to simulate colluding tamperer.
    r=output/"SOLVER_RUN_MANIFEST.json"
    obj=json.loads(r.read_text())
    obj["public_solver_pack_sha256"]=checksum(m)
    r.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="differs from canonical projection"):
        audit_solver_run(train,campaign,pack,output,source_sha=SHA)


def test_reject_external_endpoints_and_final_b_paths(tmp_path):
    train,campaign=case(tmp_path)
    pack=tmp_path/"model_only"
    prepare_solver_pack(train/"documents.jsonl",campaign,1,pack,source_sha=SHA)
    with pytest.raises(ValueError,match="local inference"):
        run_real_solver(pack,tmp_path/"bad",endpoint="https://example.org/v1",
                        model="remote",checkpoint_sha256="b"*64,source_sha=SHA)
    with pytest.raises(ValueError,match="refuse heldout/final-B"):
        prepare_solver_pack(tmp_path/"final-B160"/"documents.jsonl",campaign,1,
                            tmp_path/"bad",source_sha=SHA)
