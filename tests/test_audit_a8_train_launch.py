"""A8 CPU prelaunch provenance gates; never import or launch GPU models."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.audit_a8_train_launch import (
    A8_GROUPS, A8_TRAIN_SEED, A8_UPDATES, DEV_MANIFEST_SHA256,
    validate_a8_selection, validate_parent_adapter,
    validate_teacher_manifest,
)


def fixture_rows():
    questions = [
        {"question_id": f"train{i:02d}", "question": f"Who built the bridge {i}?",
         "gold_answer": "An engineer", "split": "train",
         "supporting_doc_ids": [f"train{i:02d}::0"]}
        for i in range(12)
    ]
    def action(qid):
        return {"question_id": qid, "student_model_gold_injected": False}
    teacher = [action(f"train{i:02d}") for i in range(8)]
    teacher.append(action("train00"))  # multiple actions per TRAIN question are valid
    heldout = [action("train08"), action("train09")]
    frozen = questions[10:11]
    return questions, teacher, heldout, frozen


def audit(source=None, teacher=None, heldout=None, frozen=None, dev=None):
    defaults = fixture_rows()
    values = [defaults[i] if x is None else x for i, x in enumerate(
        (source, teacher, heldout, frozen))]
    return validate_a8_selection(
        train_rows=values[0], teacher_train_rows=values[1],
        teacher_heldout_rows=values[2], frozen_rows=values[3],
        dev_ids=set() if dev is None else dev, seed=153, groups=2, updates=2)


def test_selection_is_stable_without_exposing_training_answers():
    result = audit()
    rev = fixture_rows()
    backwards = audit(source=list(reversed(rev[0])),
                      teacher=list(reversed(rev[1])))
    assert result == backwards
    assert result["selected_optimizer_train_questions"] == 4
    assert result["selected_counts_by_update"] == [2, 2]
    assert result["selected_frozen_overlap"] == 0
    assert result["selected_teacher_heldout_overlap"] == 0
    assert result["student_model_gold_injected"] is False
    text = json.dumps(result)
    assert "An engineer" not in text
    assert "bridge" not in text


def test_frozen_training_leak_is_rejected():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="teacher TRAIN touched frozen"):
        audit(teacher=rows[1]+[{"question_id": "train10",
                               "student_model_gold_injected": False}])


def test_teacher_heldout_isolation():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="heldout IDs overlap"):
        audit(heldout=rows[2]+[rows[1][0]])


def test_teacher_holdout_and_train64_dont_overlap():
    rows=fixture_rows()
    with pytest.raises(ValueError, match="teacher HELDOUT touched frozen"):
        audit(heldout=rows[2]+[{"question_id":"train10",
                               "student_model_gold_injected":False}])


def test_gold_visible_to_student_is_rejected():
    rows=fixture_rows()
    bad=[dict(x) for x in rows[1]]
    bad[0]["student_model_gold_injected"]=True
    with pytest.raises(ValueError, match="gold isolation"):
        audit(teacher=bad)


def test_missing_source_labels_and_non_train_split_rejected():
    source,_,_,_=fixture_rows()
    bad=[dict(x) for x in source]
    bad[0]["gold_answer"]=None
    with pytest.raises(ValueError, match="invalid labels"):
        audit(source=bad)
    bad2=[dict(x) for x in source]
    bad2[0]["split"]="dev"
    with pytest.raises(ValueError, match="non-TRAIN"):
        audit(source=bad2)


def test_dev_id_overlap_rejected_even_when_teacher_did_not_use_it():
    with pytest.raises(ValueError,match="intersects frozen DEV"):
        audit(dev={"train11"})


def test_parent_adapter_must_match_pinned_content(tmp_path):
    (tmp_path/"adapter_config.json").write_text('{"type":"test"}')
    (tmp_path/"adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="SHA drift"):
        validate_parent_adapter(tmp_path)
    (tmp_path/"adapter_model.safetensors").unlink()
    with pytest.raises(ValueError,match="missing"):
        validate_parent_adapter(tmp_path)


def test_teacher_manifest_cannot_mutate_pre_registered_tools(tmp_path):
    teacher = {
        "stage":"TRAIN_ONLY_GOLD_SUPERVISED_TEACHER_NO_RL",
        "teacher_uses_train_gold":True,
        "student_model_sees_gold_answer":False,
        "includes_only_real_strict_successes":True,
        "train_question_sha256":"fake-train-sha",
        "dev_manifest_sha256":"fake-dev-sha",
        "answer_proposer":"v4",
        "step_budget":6,
        "tool_budget":5,
    }
    p=tmp_path/"TEACHER_SUMMARY.json"
    p.write_text(json.dumps(teacher))
    assert validate_teacher_manifest(
        tmp_path,train_sha="fake-train-sha",
        dev_sha="fake-dev-sha")["model_policy_receives_gold"] is False
    teacher["tool_budget"]=7
    p.write_text(json.dumps(teacher))
    with pytest.raises(ValueError,match="preregistration drift"):
        validate_teacher_manifest(
            tmp_path,train_sha="fake-train-sha",dev_sha="fake-dev-sha")


def test_launch_constants_are_explicitly_frozen():
    assert (A8_TRAIN_SEED,A8_GROUPS,A8_UPDATES)==(153,20,2)
    assert len(DEV_MANIFEST_SHA256)==64
