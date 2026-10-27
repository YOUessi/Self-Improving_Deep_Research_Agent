"""E-030 cross-workstream exposure and prospective cohort safety tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sidra.holdout_history import (
    _source_ids, exposure_report, read_registry_verified, scan_exposures,
)
from sidra.semantic_contrast import file_sha256


def write_jsonl(path:Path, rows:list[dict])->Path:
    path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
    return path


def build_inputs(tmp_path):
    files={
        "prior_train_jsonl":[write_jsonl(tmp_path/"old_sft.jsonl",[
            {"question_id":"q1"},{"qid":"q2"},{"trace":{"question_id":"q1"}},
        ])],
        "teacher_actions":[write_jsonl(tmp_path/"teacher.jsonl",[
            {"question_id":"q3"}])],
        "optimizer_summary":[tmp_path/"optimizer.json"],
        "rsi_solver":[write_jsonl(tmp_path/"rsi_solver.jsonl",[
            {"source_train_question_id":"q4"}])],
        "rsi_reserved":[write_jsonl(tmp_path/"rsi_hold.jsonl",[
            {"original_train_question_id":"q5"}])],
        "registry":[write_jsonl(tmp_path/"old_registry.jsonl",[
            {"claim_id":"blind-old","source_train_question_id":"q6"}])],
    }
    files["optimizer_summary"][0].write_text(json.dumps({
        "selected_training_ids":["q2","q7"]}))
    return files


def frozen_packet(tmp_path):
    base=tmp_path/"frozen";base.mkdir()
    ids=["blind-one","blind-two"]
    private=write_jsonl(base/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl",[
        {"claim_id":"blind-one","source_train_question_id":"q1"},
        {"claim_id":"blind-two","source_train_question_id":"q8"},
    ])
    card_fields={
        "question":"Who wrote X?", "proposed_answer":"A",
        "provided_evidence":[{"title":"W","source_sentence_index":0,
                              "quoted_sentence":"A wrote X."}],
        "rater_instruction":"Blind human review only",
        "allowed_labels":["supported","contradicted","insufficient_evidence","unclear"],
    }
    a=write_jsonl(base/"REVIEWER_A_BLIND.jsonl",[
        {"claim_id":x, **card_fields} for x in ids])
    b=write_jsonl(base/"REVIEWER_B_BLIND.jsonl",[
        {"claim_id":x, **card_fields} for x in reversed(ids)])
    manifest={
        "split":"train", "question_count":2,
        "independent_human_adjudication_completed":False,
        "allow_as_verifier_threshold_calibration":False,
        "private_registry_sha256":file_sha256(private),
        "reviewer_packet_sha256":{"A":file_sha256(a),"B":file_sha256(b)},
        "frozen_claim_ids_sha256":hashlib.sha256(
            "\n".join(sorted(ids)).encode()).hexdigest(),
    }
    (base/"MANIFEST.json").write_text(json.dumps(manifest))
    return base


def test_read_all_six_sources_and_union_is_deduplicated(tmp_path):
    inp=build_inputs(tmp_path)
    used, per_kind, hashes=scan_exposures(
        valid_train_ids={f"q{i}" for i in range(1,10)},inputs=inp)
    assert used=={f"q{i}" for i in range(1,8)}
    assert per_kind["prior_train_jsonl"]=={"q1","q2"}
    assert per_kind["optimizer_summary"]=={"q2","q7"}
    assert len(hashes)==6
    assert all(len(x["sha256"])==64 for x in hashes.values())


def test_report_detects_actual_model_fit_overlap_not_just_old_sources(tmp_path):
    inp=build_inputs(tmp_path);frozen=frozen_packet(tmp_path)
    report, used=exposure_report(
        frozen_directory=frozen,valid_train_ids={f"q{i}" for i in range(1,10)},
        inputs=inp)
    assert len(used)==7
    assert report["frozen_cohort_size"]==2
    assert report["frozen_cohort_any_exposure_count"]==1
    assert report["frozen_cohort_any_exposure_rate"]==0.5
    assert not report["audit_pass_disjointness"]
    assert report["old_frozen_cohort_invalid_for_independent_verifier_accuracy"]
    assert report["frozen_cohort_overlap_by_source_kind"]["prior_train_jsonl"]==1
    assert report["frozen_cohort_overlap_by_source_kind"]["optimizer_summary"]==0


def test_missing_source_class_is_refused(tmp_path):
    sources=build_inputs(tmp_path)
    sources["rsi_solver"]=[]
    with pytest.raises(ValueError,match="every source class"):
        scan_exposures(valid_train_ids={"q1","q2"},inputs=sources)


def test_unknown_dev_or_test_qids_are_refused(tmp_path):
    sources=build_inputs(tmp_path)
    sources["rsi_solver"][0].write_text(
        '{"source_train_question_id":"dev-001"}\n')
    with pytest.raises(ValueError,match="unknown/non-TRAIN"):
        scan_exposures(
            valid_train_ids={f"q{i}" for i in range(1,10)},inputs=sources)


def test_duplicate_path_and_conflicting_prior_fields_fail_closed(tmp_path):
    sources=build_inputs(tmp_path)
    sources["teacher_actions"]=[sources["prior_train_jsonl"][0]]
    with pytest.raises(ValueError,match="duplicate source path"):
        scan_exposures(
            valid_train_ids={f"q{i}" for i in range(1,10)},inputs=sources)
    p=write_jsonl(tmp_path/"conflict.jsonl",[
        {"question_id":"q1","qid":"q2"}])
    with pytest.raises(ValueError,match="conflicting"):
        _source_ids(p,"prior_train_jsonl")


def test_optimizer_requires_real_selected_training_ids(tmp_path):
    file=tmp_path/"train.json"
    file.write_text('{"num_updates":2,"selected_training_ids":[]}')
    with pytest.raises(ValueError,match="missing selected_training_ids"):
        _source_ids(file,"optimizer_summary")


@pytest.mark.parametrize("tamper",["registry","reviewer","ids","split"])
def test_frozen_packet_tampering_blocks_evaluation(tmp_path,tamper):
    p=frozen_packet(tmp_path)
    if tamper=="registry":
        (p/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl").write_text("[]\n")
    elif tamper=="reviewer":
        (p/"REVIEWER_B_BLIND.jsonl").write_text("{}\n")
    elif tamper=="ids":
        registry=p/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl"
        rows=[json.loads(x) for x in registry.read_text().splitlines()]
        rows[1]["source_train_question_id"]="dev-unknown"
        write_jsonl(registry,rows)
        man=p/"MANIFEST.json";obj=json.loads(man.read_text())
        obj["private_registry_sha256"]=file_sha256(registry)
        man.write_text(json.dumps(obj))
    else:
        man=p/"MANIFEST.json";obj=json.loads(man.read_text());obj["split"]="dev"
        man.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        read_registry_verified(p,{f"q{i}" for i in range(1,10)})


def test_refreeze_must_exclude_every_case_from_old_cohort_even_if_not_trained(
    tmp_path, monkeypatch
):
    from sidra.holdout_history import make_refreeze
    from sidra import blind_adjudication
    sources=build_inputs(tmp_path)
    old=frozen_packet(tmp_path)
    def validate_exposed(**kwargs):
        derived=kwargs["prior_train_action_files"][0]
        qids={json.loads(l)["question_id"]
              for l in derived.read_text().splitlines()}
        # Old q8 was not used by any trainer: it still cannot be recycled.
        assert "q8" in qids
        assert "q1" in qids
        assert set(f"q{i}" for i in range(1,8)) <= qids
        raise RuntimeError("intentional stub, no output packet")
    monkeypatch.setattr(blind_adjudication,"freeze_blind_packet",validate_exposed)
    with pytest.raises(RuntimeError,match="intentional stub"):
        make_refreeze(
            old=old,valid_train_ids={f"q{i}" for i in range(1,10)},
            inputs=sources,dataset=tmp_path,original_train=tmp_path/"none",
            e026_rows=tmp_path/"none_e026",out=tmp_path/"new_packet")


@pytest.mark.parametrize("disjoint,expected_exit", [(True,0),(False,2)])
def test_audit_only_cli_is_read_only_and_fails_closed(
    tmp_path, monkeypatch, capsys, disjoint, expected_exit
):
    import sys
    from scripts import m7_audit_cross_workstream_holdout as cli
    source=tmp_path/"original_train_1024.json"
    source.write_text(json.dumps([
        {"_id":f"TRAIN-{i:04d}"} for i in range(1024)]))
    called=[]
    def fake_audit(**kwargs):
        called.append(kwargs)
        return {
            "audit_pass_disjointness":disjoint,
            "frozen_cohort_any_exposure_count":0 if disjoint else 1,
            "historical_source_sha256":{"/private/path":{"sha256":"abcdef"}},
        }, set()
    monkeypatch.setattr(cli,"exposure_report",fake_audit)
    args=[
        "m7_audit_cross_workstream_holdout.py",
        "--audit-only", "--frozen",str(tmp_path/"frozen"),
        "--dataset",str(tmp_path/"train"),
        "--original-train",str(source),
        "--prior-e026-rows",str(tmp_path/"e026.jsonl"),
    ]
    for flag in (
        "--prior-train-jsonl", "--teacher-actions", "--optimizer-summary",
        "--rsi-solver", "--rsi-reserved", "--obsolete-registry",
    ):
        args += [flag, str(tmp_path/(flag[2:]+".jsonl"))]
    monkeypatch.setattr(sys,"argv",args)
    if expected_exit:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code==expected_exit
    else:
        cli.main()
    stdout=capsys.readouterr().out
    assert "audit_pass_disjointness" in stdout
    assert "/private/path" not in stdout
    assert len(called)==1
    assert not (tmp_path/"new").exists()
