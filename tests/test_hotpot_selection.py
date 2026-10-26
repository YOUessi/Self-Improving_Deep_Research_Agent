"""Stable sample selection with no dependence on ground-truth labels."""
import json
import pytest
from scripts.select_hotpotqa import select


def test_selection_stable_with_reordering_and_label_changes(tmp_path):
    data = [{"_id":f"id-{i:03d}", "answer":f"answer-{i}",
             "context":[["doc",["fact"]]],"question":f"q{i}","supporting_facts":[]} for i in range(16)]
    src = tmp_path / "all.json"
    src.write_text(json.dumps(data))
    a = select(src, tmp_path / "a.json", n=7, seed=23)
    for row in data: row["answer"] = "changed-label"
    src.write_text(json.dumps(list(reversed(data))))
    b = select(src, tmp_path / "b.json", n=7, seed=23)
    assert a["selected_ids"] == b["selected_ids"]
    assert a["source_sha256"] != b["source_sha256"]
    with pytest.raises(FileExistsError):
        select(src, tmp_path / "b.json", n=7, seed=23)
