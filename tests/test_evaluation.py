import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sidra.agent import SearchAgent
from sidra.experiment import evaluate
from sidra.io import load_dataset, save_jsonl
from sidra.metrics import score_trace, summarize, token_f1
from sidra.policies import FixtureMultiHopPolicy
from sidra.schema import Action, Document, Question
from sidra.sft import build_sft_examples


def test_token_f1_not_exact_match():
    assert 0.0 < token_f1("red blue", "red green") < 1.0


def test_end_to_end_trace_manifest_and_train_export(fixture_dir, tmp_path):
    run = tmp_path / "run_train"
    summary = evaluate(fixture_dir, "train", FixtureMultiHopPolicy(), run, limit=2)
    assert summary["n"] == 2
    assert summary["verified_success_mean"] == 1
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["stage"] == "smoke_fixture"
    assert manifest["no_gpu_training_claimed"] is True
    assert len(manifest["data_sha256"]["documents.jsonl"]) == 64
    output = tmp_path / "sft.jsonl"
    report = build_sft_examples(fixture_dir, run / "traces.jsonl", output)
    assert report["action_training_examples"] == 10
    examples = [json.loads(x) for x in output.read_text().splitlines()]
    assert "gold_answer" not in json.dumps(examples)
    assert all(set(e) == {"prompt", "completion", "question_id", "step"} for e in examples)
    with pytest.raises(FileExistsError):
        evaluate(fixture_dir, "train", FixtureMultiHopPolicy(), run, limit=2)


def test_reject_dev_as_sft_source(fixture_dir, tmp_path):
    run = tmp_path / "dev"
    evaluate(fixture_dir, "dev", FixtureMultiHopPolicy(), run, limit=1)
    with pytest.raises(ValueError, match="Cannot generate supervised training actions from dev"):
        build_sft_examples(fixture_dir, run / "traces.jsonl", tmp_path / "bad.jsonl")
    assert not (tmp_path / "bad.jsonl").exists()


def test_dataset_group_leakage_rejected(tmp_path):
    save_jsonl(tmp_path / "documents.jsonl", [asdict(Document("doc", "title", "fact"))])
    save_jsonl(tmp_path / "questions.jsonl", [
        {**asdict(Question("a", "a", "a", split="train", group="group")), "supporting_doc_ids": []},
        {**asdict(Question("b", "b", "b", split="dev", group="group")), "supporting_doc_ids": []},
    ])
    with pytest.raises(ValueError, match="Group leakage"):
        load_dataset(tmp_path)


def test_verifier_does_not_credit_correct_answer_without_evidence(fixture_data):
    docs, questions = fixture_data
    q = questions[0]
    result = SearchAgent(docs, FixtureMultiHopPolicy()).run(q)
    result.citations = []
    assert score_trace(q, result, {d.doc_id: d for d in docs})["verified_success"] == 0
