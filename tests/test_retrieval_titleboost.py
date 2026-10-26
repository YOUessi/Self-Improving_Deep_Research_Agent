"""Title boost changes ranking only, never train/gold-conditioned scores."""
import pytest

from scripts.sweep_titleboost_pilot import score_retriever, sweep
from sidra.retrieval import BM25Index
from sidra.schema import Document


def test_bm25_title_boost_prefers_title_match():
    docs = [
        Document("a","Oslo","One Oslo reference"),
        Document("b","Other","Oslo Oslo Oslo Oslo news"),
    ]
    query = "Oslo"
    baseline = BM25Index(docs).search(query,k=2)
    boosted = BM25Index(docs,title_boost=8.0).search(query,k=2)
    assert baseline[0].doc_id == "b"
    assert boosted[0].doc_id == "a"


def test_invalid_bm25_boost_rejected():
    docs=[Document("a","T","Text")]
    with pytest.raises(ValueError):
        BM25Index(docs,title_boost=-1)
    with pytest.raises(ValueError):
        BM25Index(docs,title_boost=float("nan"))


def test_pilot_sweep_is_frozen(fixture_dir,tmp_path):
    from sidra.io import load_dataset, save_jsonl
    docs, questions = load_dataset(fixture_dir)
    dev = tmp_path / "dev_only"
    dev.mkdir()
    save_jsonl(dev / "documents.jsonl", [d.__dict__ for d in docs])
    save_jsonl(dev / "questions.jsonl",
               [q.__dict__ for q in questions if q.split == "dev"])
    out=tmp_path/"title.json"
    report=sweep(dev,out,boosts=(0.0,0.5,2.0))
    assert report["chosen_title_boost"] in {0.0,0.5,2.0}
    assert report["partition"].startswith("pilot only")
    assert len(report["results"])==3
    with pytest.raises(FileExistsError): sweep(dev,out)
