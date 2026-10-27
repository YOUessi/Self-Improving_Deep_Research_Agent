"""The candidate-ceiling audit is DEV pilot only and never passes gold to policy."""
import json

import pytest

from scripts.audit_finish_candidate_ceiling import examine, diagnose
from sidra.io import save_jsonl
from sidra.schema import Document, Question


def test_oracle_exposure_does_not_force_correct_finish_candidate():
    docs = {
        "a": Document("a", "A", "The answer is yes: first premise."),
        "b": Document("b", "B", "Second premise states that yes is correct."),
    }
    q = Question("q", "Is the statement true?", "yes",
                 supporting_doc_ids=("a", "b"), split="dev")
    result = examine(q, docs, proposer="v4")
    assert result["answer_occurs_in_read_text"]
    assert not result["gold_available_as_action"]
    assert result["gold_is_binary"]
    assert "gold_answer" not in result
    assert "question" not in result


def test_frozen_heldout_cannot_be_inspected_by_oracle_candidate_audit(tmp_path):
    pilot = tmp_path / "pilot" / "converted"
    pilot.mkdir(parents=True)
    docs = [
        Document("a", "Alpha", "The answer was in Acme City."),
        Document("b", "Beta", "Acme City is located on Island Green."),
    ]
    qs = [
        Question("q", "What city was the answer in?", "Acme City",
                 supporting_doc_ids=("a", "b"), split="dev")
    ]
    save_jsonl(pilot / "documents.jsonl", [d.__dict__ for d in docs])
    save_jsonl(pilot / "questions.jsonl", [q.__dict__ for q in qs])
    report = diagnose(pilot, tmp_path / "pilot_aggregate.json")
    assert report["n"] == 1
    assert report["partition"].startswith("DEV pilot ONLY")
    assert "gold_answer" not in (tmp_path / "pilot_aggregate.json").read_text()
    with pytest.raises(FileExistsError):
        diagnose(pilot, tmp_path / "pilot_aggregate.json")
    heldout = tmp_path / "heldout" / "converted"
    heldout.mkdir(parents=True)
    with pytest.raises(ValueError, match="DEV pilot"):
        diagnose(heldout, tmp_path / "cannot_inspect.json")
