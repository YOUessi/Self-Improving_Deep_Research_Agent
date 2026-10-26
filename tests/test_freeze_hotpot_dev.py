"""Freeze partitions are immutable, disjoint and independent of answers."""
import hashlib
import json

import pytest

from scripts.freeze_hotpot_dev import freeze_dev
from sidra.io import load_dataset


def source_file(tmp_path, *, change_answer=False, reverse=False):
    rows = [
        {"_id": f"qa-{i}", "question": f"What is {i}?", "answer":
         ("CHANGED" if change_answer else f"ans-{i}"),
         "context": [["source", [f"source {i}"]], ["support", [f"ans-{i}"]]],
         "supporting_facts": [["source", 0], ["support", 0]]}
        for i in range(20)
    ]
    if reverse:
        rows.reverse()
    src = tmp_path / ("changed.json" if change_answer else ("reverse.json" if reverse else "original.json"))
    src.write_text(json.dumps(rows))
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


def test_dev_freeze_is_disjoint_and_unusable_for_train(tmp_path):
    raw, sha = source_file(tmp_path)
    output = tmp_path / "split"
    info = freeze_dev(raw, output, expected_sha256=sha, seed=11, pilot=5, heldout=7)
    a = set(info["pilot"]["selected_ids"])
    b = set(info["heldout"]["selected_ids"])
    assert len(a) == 5 and len(b) == 7 and not a & b
    for split, count in (("pilot", 5), ("heldout", 7)):
        docs, questions = load_dataset(output / split / "converted")
        assert len(questions) == count
        assert all(x.split == "dev" for x in questions)
        assert info[split]["documents_sha256"] == hashlib.sha256(
            (output / split / "converted" / "documents.jsonl").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        freeze_dev(raw, output, expected_sha256=sha, seed=11, pilot=5, heldout=7)


def test_label_changes_do_not_change_selected_ids(tmp_path):
    s1, sha1 = source_file(tmp_path)
    s2, sha2 = source_file(tmp_path, change_answer=True)
    s3, sha3 = source_file(tmp_path, reverse=True)
    m1 = freeze_dev(s1, tmp_path / "a", expected_sha256=sha1, pilot=4, heldout=6)
    m2 = freeze_dev(s2, tmp_path / "b", expected_sha256=sha2, pilot=4, heldout=6)
    m3 = freeze_dev(s3, tmp_path / "c", expected_sha256=sha3, pilot=4, heldout=6)
    for k in ("pilot", "heldout"):
        assert m1[k]["selected_ids"] == m2[k]["selected_ids"] == m3[k]["selected_ids"]


def test_bad_source_hash_does_not_write_files(tmp_path):
    raw, _ = source_file(tmp_path)
    target = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="SHA256"):
        freeze_dev(raw, target, expected_sha256="f" * 64, pilot=4, heldout=6)
    assert not target.exists()
