from sidra.retrieval import BM25Index, tokenize
from sidra.schema import Document


def test_bm25_exact_entity_ranked_first(fixture_data):
    documents, _ = fixture_data
    index = BM25Index(documents)
    hits = index.search("Explorer Mara014", k=3)
    assert hits[0].doc_id == "explorer_014"
    region = index.search("Vale014", k=4)
    assert any(hit.doc_id == "region_014" for hit in region)


def test_scoped_retrieval_isolation():
    docs = [Document("a", "shared", "word", "question-a"),
            Document("b", "shared", "word", "question-b")]
    idx = BM25Index(docs)
    assert [h.doc_id for h in idx.search("word", scope="question-a")] == ["a"]
    assert [h.doc_id for h in idx.search("word", scope="question-b")] == ["b"]


def test_cjk_tokenization():
    assert tokenize("机器学习 Agent")[:4] == ["机", "器", "学", "习"]
