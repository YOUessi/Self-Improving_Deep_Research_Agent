"""Offline model verdict replay must be monotone rejection, never promotion."""
import pytest

from scripts.audit_rsi_semantic_posthoc import classify_conservatively
from sidra.rsi_semantic_triage import ReviewPacket


def packet(*,question,answer,evidence):
    return ReviewPacket(
        challenge_id="c-"+"e"*25,
        question=question,answer=answer,
        evidence=tuple(evidence),sha256="f"*64,
    )


def old(packet,*,status="needs_independent_semantic_label_review",
        bridge="Bronson"):
    return {
        "challenge_id":packet.challenge_id,
        "review_packet_sha256":packet.sha256,
        "conclusion":status,
        "model_verdicts":["supported","supported"],
        "model_bridge_entities":[bridge,bridge],
        "reviewer_output_sha256":["a"*64,"b"*64],
        "authorized_train_label_count":0,
        "trainer_access_granted":False,
    }


def test_observed_location_name_false_bridge_downgrades_without_gold():
    p=packet(
        question="What was the population of the city where the coffee company is based?",
        answer="2,349",
        evidence=[
            {"doc_id":"coffee","quote":"Acme & Bronson is an Olympia, Washington based coffee roaster."},
            {"doc_id":"city","quote":"Bronson is a city in Michigan. The population was 2,349 at census."},
        ],
    )
    r=classify_conservatively(p,old(p))
    assert r["original_conclusion"]=="needs_independent_semantic_label_review"
    assert r["revised_conclusion"]=="flagged_deterministic_premise_gap"
    assert "bridge_component_of_larger_named_entity" in r["negative_flags"]
    assert "company_base_city_not_in_population_evidence" in r["negative_flags"]
    assert r["approved_train_labels"]==0
    assert r["promoted_to_trainer"] is False


def test_missing_championship_relation_downgrades_both_order_reviews():
    p=packet(
        question="Which team did the player win an NBA championship with?",
        answer="San Antonio Spurs",
        evidence=[
            {"doc_id":"team1","quote":"The player is signed to San Antonio Spurs."},
            {"doc_id":"team2","quote":"San Antonio Spurs were based in Texas."},
        ],
    )
    r=classify_conservatively(p,old(p,bridge="San Antonio Spurs"))
    assert r["revised_conclusion"]=="flagged_deterministic_premise_gap"
    assert r["negative_flags"]==[
        "championship_relation_absent_from_both_quotes"
    ]


def test_unflagged_review_still_unlabelled_without_upgrade():
    p=packet(
        question="Which poem by Max Ehrmann was used in devotional recordings?",
        answer="Desiderata",
        evidence=[
            {"doc_id":"poem","quote":"Desiderata is a poem by Max Ehrmann used in devotional recordings."},
            {"doc_id":"bio","quote":"Max Ehrmann was an American writer of Desiderata."},
        ],
    )
    r=classify_conservatively(p,old(p,bridge="Max Ehrmann"))
    assert r["revised_conclusion"]=="needs_independent_semantic_label_review"
    assert r["approved_train_labels"]==0


def test_negative_or_tampered_artifact_cannot_become_positive():
    p=packet(
        question="Which poem by Max Ehrmann was used in devotional recordings?",
        answer="Desiderata",
        evidence=[
            {"doc_id":"poem","quote":"Desiderata is a poem by Max Ehrmann used in devotional recordings."},
            {"doc_id":"bio","quote":"Max Ehrmann wrote Desiderata in 1927."},
        ],
    )
    negative=old(p,status="flagged_unsupported",bridge="Max Ehrmann")
    r=classify_conservatively(p,negative)
    assert r["original_conclusion"]==r["revised_conclusion"]=="flagged_unsupported"
    negative["trainer_access_granted"]=True
    with pytest.raises(ValueError,match="untrusted"):
        classify_conservatively(p,negative)
