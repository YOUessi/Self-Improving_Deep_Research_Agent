"""Regression: a multi-agent lexical Verifier must never invent citations."""
from sidra.evidence_audit import EvidenceAudit
from sidra.schema import Action


def decide(answer,obs):
    return EvidenceAudit(min_citations=2).verify(
        Action("finish",answer=answer,citations=("doc-a","doc-b")),obs)


def test_cross_document_boundary_does_not_fabricate_new_york():
    obs=[
        {"kind":"read","doc_id":"doc-a","text":"The city is called New"},
        {"kind":"read","doc_id":"doc-b","text":"York was a writer"},
    ]
    v=decide("New York",obs)
    assert v.approved is False and v.reason=="answer_not_in_sources"
    # The actual proof must occur contiguously inside a single READ.
    obs[1]["text"]="The city of New York is in the United States."
    v=decide("New York",obs)
    assert v.approved is True
    assert v.reason=="lexical_only"  # NOT semantic entailment!


def test_answer_word_boundary_not_just_a_substring():
    obs=[
        {"kind":"read","doc_id":"doc-a","text":"Yorkshire is in England."},
        {"kind":"read","doc_id":"doc-b","text":"A New Yorker visited."},
    ]
    assert decide("York",obs).approved is False
    obs[0]["text"]="York is a city in England."
    assert decide("York",obs).approved is True


def test_unread_and_duplicate_citations_fail_closed():
    verifier=EvidenceAudit(min_citations=2)
    obs=[{"kind":"read","doc_id":"doc-a","text":"New York"}]
    assert verifier.verify(
        Action("finish",answer="New York",citations=("doc-a","doc-b")),obs
    ).reason=="unread_citations"
    assert verifier.verify(
        Action("finish",answer="New York",citations=("doc-a","doc-a")),obs
    ).reason=="missing_or_duplicate_citations"


def test_negation_remains_lexical_only_and_requires_NLI_in_separate_gate():
    obs=[
        {"kind":"read","doc_id":"doc-a","text":"Jane was not born in Warsaw."},
        {"kind":"read","doc_id":"doc-b","text":"Warsaw is a city."},
    ]
    v=decide("Warsaw",obs)
    assert v.approved and v.reason=="lexical_only"
    # Intentionally do not treat this as a true semantic approval.
