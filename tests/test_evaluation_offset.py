"""Frozen dev position slicing keeps early diagnostic questions out of later tests."""
import json
from sidra.experiment import evaluate
from sidra.io import load_dataset
from sidra.policies import FixtureMultiHopPolicy


def test_offset_selects_new_question_ids(fixture_dir, tmp_path):
    _, questions = load_dataset(fixture_dir)
    qids = sorted(q.question_id for q in questions if q.split == "dev")
    destination = tmp_path / "offset2"
    summary = evaluate(fixture_dir, "dev", FixtureMultiHopPolicy(),
                       destination, limit=2, seed=42, offset=2)
    assert summary["n"] == 2
    rows = [json.loads(x) for x in (destination / "per_example.jsonl").read_text().splitlines()]
    assert [x["question_id"] for x in rows] == qids[2:4]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["evaluation_offset"] == 2
