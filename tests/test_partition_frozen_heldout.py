"""Partition heldout independently of labels and preserve the hash order."""
import hashlib
import json

import pytest

from scripts.freeze_hotpot_dev import freeze_dev
from scripts.partition_frozen_heldout import partition


def test_eval_a_and_final_b_are_immutable_disjoint_dev(tmp_path):
    rows=[
        {"_id": f"id-{i}", "question": f"Where {i}", "answer":str(i),
         "context":[["A",["fact A"]],["B",["fact B"]]],
         "supporting_facts":[["A",0],["B",0]]}
        for i in range(16)]
    src=tmp_path/"input.json"
    src.write_text(json.dumps(rows))
    sha=hashlib.sha256(src.read_bytes()).hexdigest()
    frozen=tmp_path/"frozen"
    main=freeze_dev(src,frozen,expected_sha256=sha,pilot=4,heldout=12)
    dest=tmp_path/"heldout_splits"
    x=partition(frozen,dest,eval_a=5)
    assert x["eval_a_size"]==5 and x["sealed_final_b_size"]==7
    assert x["eval_a"]["ids"]==main["heldout"]["selected_ids"][:5]
    assert x["final_b"]["ids"]==main["heldout"]["selected_ids"][5:]
    assert not set(x["eval_a"]["ids"]) & set(x["final_b"]["ids"])
    assert not set(main["pilot"]["selected_ids"]) & set(x["eval_a"]["ids"])
    assert "train" not in (dest/"final_b/converted/questions.jsonl").read_text()
    with pytest.raises(FileExistsError):
        partition(frozen,dest,eval_a=5)


def test_partition_rejects_modified_source(tmp_path):
    rows=[
        {"_id": f"x-{i}", "question":"a?", "answer":"b",
         "context":[["A",["x"]],["B",["y"]]],"supporting_facts":[["A",0],["B",0]]}
        for i in range(9)]
    raw=tmp_path/"src.json";raw.write_text(json.dumps(rows))
    frozen=tmp_path/"freeze"
    freeze_dev(raw,frozen,expected_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),pilot=2,heldout=6)
    with (frozen/"heldout/original_dev_subset.json").open("a") as f:f.write(" ")
    with pytest.raises(ValueError,match="modified"):
        partition(frozen,tmp_path/"should_not_exist",eval_a=3)
