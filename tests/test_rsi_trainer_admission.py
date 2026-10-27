"""TRAIN-only multi-cohort RSI candidate handoff refuses unverified labels."""
import hashlib
import json

import pytest

from sidra.rsi_trainer_admission import (
    Cohort,audit_training_admission,load_rows,sha,
)


def data(tmp_path,*,split="train"):
    directory=tmp_path/"train"
    directory.mkdir()
    questions=[
        {"question_id":f"train{i}","question":f"Which place has relationship {i} with the person?",
         "gold_answer":"Alpha","supporting_doc_ids":["doc1","doc2"],
         "scope":"global","split":split}
        for i in range(4)
    ]
    docs=[
        {"doc_id":"doc1","title":"A","text":"Alpha is a town."},
        {"doc_id":"doc2","title":"B","text":"X is related to Alpha."},
    ]
    for name,items in (("questions.jsonl",questions),("documents.jsonl",docs)):
        (directory/name).write_text(
            "\n".join(json.dumps(x) for x in items)+"\n")
    return directory


def fake_cohort(generation:int,seed:int,source_index:int, *,
                status:str="flagged_unsupported",proposal:str=""):
    m={
        "cohort_index":generation,"sampling_seed":seed,
        "quote_valid_n":int(bool(proposal)),
        "pending_independent_labels_n":int(status=="needs_independent_semantic_label_review"),
        "status_counts":{status:1},
    }
    row={
        "challenge_id":f"c-{generation:024d}",
        "source_train_id":f"train{source_index}",
        "source_question_text":
            f"Which place has relationship {source_index} with the person?",
        "tier":"bridge","status":status,
        "candidate_question_text":proposal,
    }
    return m,[row]


def cohorts():
    return [Cohort(*([__import__("pathlib").Path(f"x{i}")] * 4)) for i in (1,2)]


def install_replay(monkeypatch,stage_results):
    import sidra.rsi_trainer_admission as mod
    sequence=iter(stage_results)
    monkeypatch.setattr(mod,"review_cohort",lambda *_: next(sequence))


def test_empty_trainer_export_is_stable_and_immutable(tmp_path,monkeypatch):
    directory=data(tmp_path)
    install_replay(monkeypatch,[
        fake_cohort(1,43,0,status="flagged_unsupported"),
        fake_cohort(2,44,1,status="needs_independent_semantic_label_review",
                    proposal="Who else worked with the person in a different city?"),
    ])
    out=tmp_path/"out"
    result=audit_training_admission(directory,cohorts(),out)
    assert result["n_distinct_train_origins"]==2
    assert result["n_semantic_pending"]==1
    assert result["independent_verified_novel_labels"]==0
    assert not result["multi_generation_RSI_completed"]
    assert result["trainer_input_rows"]==0
    assert (out/"TRAINER_INPUT.jsonl").read_bytes()==b""
    assert sha(out/"TRAINER_INPUT.jsonl")==hashlib.sha256(b"").hexdigest()
    assert result["cohorts"][0]["sampling_seed"]==43
    assert "train0" not in (out/"ADMISSION_AUDIT.json").read_text()
    with pytest.raises(FileExistsError):
        audit_training_admission(directory,cohorts(),out)


@pytest.mark.parametrize("case,expected",[
    ("duplicate_source","duplicate"),
    ("duplicate_challenge","duplicate"),
    ("same_seed","reused seed"),
    ("same_generation","noncontiguous"),
    ("wrong_source_question","original TRAIN question"),
    ("near_duplicate","near-duplicate"),
])
def test_no_cross_cohort_collision_or_source_forgery(tmp_path,monkeypatch,case,expected):
    directory=data(tmp_path)
    first=fake_cohort(1,43,0,proposal=
        "What city is associated with the painter of the bronze medallion?")
    second=fake_cohort(2,44,1,proposal=
        "Which artist composed the melody for the North Sea celebration?")
    a,b=first,second
    if case=="duplicate_source":b[1][0]["source_train_id"]="train0"
    if case=="duplicate_challenge":b[1][0]["challenge_id"]=a[1][0]["challenge_id"]
    if case=="same_seed":b[0]["sampling_seed"]=43
    if case=="same_generation":b[0]["cohort_index"]=1
    if case=="wrong_source_question":b[1][0]["source_question_text"]="falsified"
    if case=="near_duplicate":
        b[1][0]["candidate_question_text"]=a[1][0]["candidate_question_text"]
    install_replay(monkeypatch,[a,b])
    with pytest.raises(ValueError,match=expected):
        audit_training_admission(directory,cohorts(),tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_non_train_or_sealed_path_rejected_without_reading_labels(tmp_path):
    bad=data(tmp_path,split="dev")
    with pytest.raises(ValueError,match="only TRAIN"):
        audit_training_admission(bad,cohorts(),tmp_path/"out")
    sealed=tmp_path/"final-B160"
    sealed.mkdir()
    with pytest.raises(ValueError,match="sealed"):
        audit_training_admission(sealed,cohorts(),tmp_path/"out")


def test_jsonl_parse_rejects_blank_or_nonobject(tmp_path):
    p=tmp_path/"malformed.jsonl"
    p.write_text('{"x":1}\n\n')
    with pytest.raises(ValueError,match="blank"):
        load_rows(p)
    p.write_text("[]\n")
    with pytest.raises(ValueError,match="non-object"):
        load_rows(p)
