"""CPU-only source reconstruction catches freeze tampering and past TRAIN exposure."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.audit_a8_frozen_integrity import (
    FreezeProtocol, select_disjoint_train, validate_a8_freeze, _sha256,
)


def _dump(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))


def _save(path:Path,row:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(row,sort_keys=True)+"\n")


def _fixture(root:Path):
    source=root/"source"; frozen=root/"frozen"; m7=root/"m7"
    original=[{
        "question_id":f"train-{i:02d}",
        "question":f"How many pages in document {i}?",
        "gold_answer":str(i*3),"supporting_doc_ids":[f"doc-{i}"],
        "split":"train","scope":"global",
    } for i in range(6)]
    docs=[{"doc_id":f"doc-{i}","title":str(i),"text":"evidence"}
          for i in range(6)]
    _dump(source/"questions.jsonl",original)
    _dump(source/"documents.jsonl",docs)
    _dump(frozen/"documents.jsonl",docs)
    excluded={"train-00","train-01"}
    ordered=select_disjoint_train(
        {r["question_id"] for r in original},excluded,seed=919,count=2)
    _dump(frozen/"questions.jsonl",
          sorted((x for x in original if x["question_id"] in ordered),
                 key=lambda r:r["question_id"]))
    teacher_rel="data/original/TEACHER_ACTIONS_TRAIN.jsonl"
    fit_rel="results/original/TRAIN_SUMMARY.json"
    _dump(m7/teacher_rel,[{"question_id":"train-00"}])
    _save(m7/fit_rel,{
        "stage":"real_online_finite_constrained_categorical_grpo_style",
        "split":"train","dev_gold_used_for_reward":False,
        "selected_training_ids":["train-01"],
    })
    teacher={teacher_rel:_sha256(m7/teacher_rel)}
    fit={fit_rel:_sha256(m7/fit_rel)}
    proto=FreezeProtocol(
        source_questions_sha256=_sha256(source/"questions.jsonl"),
        source_documents_sha256=_sha256(source/"documents.jsonl"),
        frozen_questions_sha256=_sha256(frozen/"questions.jsonl"),
        frozen_documents_sha256=_sha256(frozen/"documents.jsonl"),
        ordered_qid_sha256=hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        count=2,excluded_count=2,selection_seed=919,source_sha="a"*40,
    )
    man={field:getattr(proto,field) for field in proto.__dataclass_fields__}
    man.update({
        "split":"train","model_training_labels_allowed":False,
        "candidate_count":4,"teacher_files_checked":1,
        "prior_fit_files_checked":1,
    })
    _save(frozen/"FREEZE_MANIFEST.json",man)
    return m7,source,frozen,proto,teacher,fit


def _audit(fixture):
    m7,source,frozen,proto,teacher,fit=fixture
    return validate_a8_freeze(
        m7root=m7,source=source,frozen=frozen,
        protocol=proto,teacher_history=teacher,fit_history=fit)


def test_frozen_source_replay_checks_history_and_names(tmp_path):
    fixture=_fixture(tmp_path)
    report=_audit(fixture)
    assert report["status"]=="PASS"
    assert report["eligible_count"]==4
    assert report["frozen_count"]==2
    assert report["teacher_files_pinned"]==report["training_summaries_pinned"]==1
    assert not report["sealed_final_B160_opened"]


def test_source_or_freeze_mutation_always_rejected(tmp_path):
    fixture=_fixture(tmp_path)
    _,source,frozen,_,_,_ = fixture
    p=frozen/"questions.jsonl"
    p.write_text(p.read_text().replace("How many", "How MANY"))
    with pytest.raises(ValueError,match="checksum"):
        _audit(fixture)
    fixture=_fixture(tmp_path)
    (source/"documents.jsonl").write_text("[]\n")
    with pytest.raises(ValueError,match="checksum"):
        _audit(fixture)


def test_historical_teacher_and_optimizer_summary_tampering_rejected(tmp_path):
    fixture=_fixture(tmp_path)
    m7,_,_,_,teacher,fit=fixture
    p=m7/next(iter(teacher))
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError,match="history input"):
        _audit(fixture)
    fixture=_fixture(tmp_path)
    m7,_,_,_,teacher,fit=fixture
    p=m7/next(iter(fit))
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError,match="history input"):
        _audit(fixture)


def test_reconstruct_selection_refuses_wrong_id_set(tmp_path):
    fixture=_fixture(tmp_path)
    m7,source,frozen,p,t,f =fixture
    wrong=replace(p,ordered_qid_sha256="f"*64)
    with pytest.raises(ValueError,match="ordered"):
        validate_a8_freeze(m7root=m7,source=source,frozen=frozen,
                            protocol=wrong,teacher_history=t,fit_history=f)
    with pytest.raises(ValueError,match="unknown TRAIN"):
        select_disjoint_train({"train-1"},{"dev-1"},919,1)


def test_later_train_fit_contamination_detected_without_recomputing_freeze(tmp_path):
    fixture=_fixture(tmp_path)
    m7,source,frozen,_,_,_=fixture
    selected=json.loads((frozen/"questions.jsonl").read_text().splitlines()[0])
    _save(m7/"results"/"later"/"TRAIN_SUMMARY.json",{
        "stage":"real_online_finite_constrained_categorical_grpo_style",
        "split":"train","selected_training_ids":[selected["question_id"]],
    })
    with pytest.raises(ValueError,match="contaminated by later"):
        _audit(fixture)


def test_train_dev_leakage_rejected_even_if_checksums_are_resigned(tmp_path):
    fixture=_fixture(tmp_path)
    m7,source,frozen,p,t,f=fixture
    rows=[json.loads(x) for x in (source/"questions.jsonl").read_text().splitlines()]
    rows[0]["split"]="dev"
    _dump(source/"questions.jsonl",rows)
    modified=replace(p,source_questions_sha256=_sha256(source/"questions.jsonl"))
    with pytest.raises(ValueError,match="TRAIN-only"):
        validate_a8_freeze(m7root=m7,source=source,frozen=frozen,
                            protocol=modified,teacher_history=t,fit_history=f)
