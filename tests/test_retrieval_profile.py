"""Freeze sample selection and never use labels to rank search results."""

import json
import pytest

from sidra.schema import Question
from scripts.profile_hotpot_retrieval import frozen_selection, evaluate_retrieval


def test_frozen_subset_independent_of_input_order():
    questions = [Question(str(i), "Text", "Answer", split="dev") for i in range(31)]
    a = frozen_selection(questions, limit=8, seed=42)
    b = frozen_selection(list(reversed(questions)), limit=8, seed=42)
    assert [x.question_id for x in a] == [x.question_id for x in b]


def test_retrieval_diagnostic_runs_on_synthetic_fixture(fixture_dir, tmp_path):
    out = tmp_path / "report"
    stats = evaluate_retrieval(fixture_dir, out, limit=3, seed=42, ks=(1, 3))
    assert stats["count"] == 3
    assert stats["scope"].startswith("per-question")
    assert 0 <= stats["support_recall_at_1"] <= 1
    assert (out / "per_example.jsonl").is_file()
    with pytest.raises(FileExistsError):
        evaluate_retrieval(fixture_dir, out, limit=3)
