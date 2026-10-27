"""Adversarial semantic triage: no machine response can create a TRAIN label."""
import json

import pytest

from sidra.rsi_semantic_triage import (
    LocalSemanticReviewer,build_review_packet,consensus,messages,
    verify_review_response,
)


def candidate(*,question="What poem by M. Vale appeared in audio readings?",
              answer="River Plain",
              quotes=None):
    if quotes is None:
        quotes=[
            {"doc_id":"doc-a","quote":"The River Plain was written by the poet M. Vale in 1952."},
            {"doc_id":"doc-b","quote":"M. Vale was born in June 1908; his poem River Plain was in audio readings."},
        ]
    return {
        "challenge_id":"c-"+"a"*25,
        "question":question,"proposed_answer":answer,
        "reason":"untrusted challenger said it follows from the sources",
        "evidence":quotes,
        "verdict":"quote_extracted_unlabelled_pending_semantic_verification",
        "packet_sha256":"b"*64,"completion_sha256":"c"*64,
    }


def supported(*,bridge="M. Vale",proof=None):
    if proof is None:proof=candidate()["evidence"]
    return json.dumps({
        "verdict":"supported","bridge_entity":bridge,
        "bridge_type":"person","proof":proof,"discrepancy":"",
    })


def negative(what="unsupported",why="The shared name does not identify the same city."):
    return json.dumps({
        "verdict":what,"bridge_entity":"","bridge_type":"",
        "proof":[],"discrepancy":why,
    })


def test_model_view_never_exposes_challenger_reason_or_train_gold():
    c=candidate()
    c["reason"]="TRAIN GOLD secret! source TRAIN q.gold_answer=hidden"
    p=build_review_packet(c)
    user=json.loads(messages(p)[1]["content"])
    assert set(user)=={"question","proposed_answer","source_quotes"}
    assert "secret" not in json.dumps(user)
    assert "gold_answer" not in json.dumps(user)
    assert "supporting_doc_ids" not in json.dumps(user)
    assert "TRAIN GOLD" not in json.dumps(user)


def test_two_ordered_supported_witnesses_still_never_issue_gold():
    p=build_review_packet(candidate())
    a=verify_review_response(supported(),p)
    b=verify_review_response(supported(proof=list(reversed(candidate()["evidence"]))),p)
    assert a["machine_review_status"]=="structurally_witnessed_unlabelled"
    report=consensus(p,[a,b])
    assert report["conclusion"]=="needs_independent_semantic_label_review"
    assert not report["trainer_access_granted"]
    assert report["authorized_train_label_count"]==0


def test_location_name_collision_must_not_pass_as_olympia():
    p=build_review_packet(candidate(
        question="What was the population of the town where Batdorf and Bronson operated?",
        answer="2,349",
        quotes=[
            {"doc_id":"coffee","quote":"Batdorf & Bronson is an Olympia, Washington based coffee roaster."},
            {"doc_id":"city","quote":"Bronson is a city in Michigan. The population was 2,349 at census."},
        ],
    ))
    proof=list(p.evidence)
    verdict=verify_review_response(supported(bridge="Olympia",proof=proof),p)
    assert verdict["machine_review_status"]=="invalid"
    assert verdict["reason"]=="bridge_entity_not_quoted_in_both_documents"
    rejections=[verify_review_response(negative("contradicted"),p),
                verify_review_response(negative("unsupported"),p)]
    summary=consensus(p,rejections)
    assert summary["conclusion"]=="flagged_contradiction"
    assert summary["authorized_train_label_count"]==0


def test_ambiguous_bronson_same_string_still_never_yields_label():
    p=build_review_packet(candidate(
        question="What was the population of the town where Batdorf and Bronson operated?",
        answer="2,349",
        quotes=[
            {"doc_id":"coffee","quote":"Batdorf & Bronson is an Olympia, Washington based coffee roaster."},
            {"doc_id":"city","quote":"Bronson is a city in Michigan. The population was 2,349 at census."},
        ],
    ))
    # A prompt-injected or compromised machine reviewer can hallucinate a
    # common name despite entity-type mismatch. Structural evidence is
    # necessary but NEVER sufficient for semantic truth or a training label.
    hallucination=verify_review_response(
        supported(bridge="Bronson",proof=list(p.evidence)),p)
    assert hallucination["machine_review_status"]=="structurally_witnessed_unlabelled"
    report=consensus(p,[hallucination,hallucination])
    assert report["conclusion"]=="flagged_deterministic_premise_gap"
    assert "bridge_component_of_larger_named_entity" in report["deterministic_negative_flags"]
    assert "company_base_city_not_in_population_evidence" in report["deterministic_negative_flags"]
    assert report["authorized_train_label_count"]==0


def test_absent_championship_claim_rejected_even_with_shared_team():
    c=candidate(
        question="Which NBA team did the player win a championship with?",
        answer="San Antonio Spurs",
        quotes=[
            {"doc_id":"one","quote":"The player is a member of the San Antonio Spurs team."},
            {"doc_id":"two","quote":"The San Antonio Spurs featured three basketball players."},
        ],
    )
    p=build_review_packet(c)
    r=verify_review_response(supported(bridge="San Antonio Spurs",proof=c["evidence"]),p)
    assert r["machine_review_status"]=="structurally_witnessed_unlabelled"
    report=consensus(p,[r,r])
    assert report["conclusion"]=="flagged_deterministic_premise_gap"
    assert "championship_relation_absent_from_both_quotes" in report["deterministic_negative_flags"]
    assert report["authorized_train_label_count"]==0


def test_fabricated_second_quote_is_rejected():
    p=build_review_packet(candidate())
    forged=list(p.evidence)
    forged[1]={"doc_id":"doc-b","quote":"M. Vale is the leader of the World Cricket Federation."}
    result=verify_review_response(supported(proof=forged),p)
    assert result["reason"]=="positive_proof_invented_quote"
    assert result["machine_review_status"]=="invalid"


def test_one_document_witness_or_missing_answer_is_rejected():
    p=build_review_packet(candidate())
    evidence=list(p.evidence)
    same=[evidence[0],evidence[0]]
    assert verify_review_response(supported(proof=same),p)["machine_review_status"]=="invalid"
    fabricated=[{"doc_id":"doc-a","quote":"The River Plain was written by the poet M. Vale in 1952."},
                {"doc_id":"doc-b","quote":"A completely unrelated fact from nowhere recorded as evidence."}]
    assert verify_review_response(supported(proof=fabricated),p)["machine_review_status"]=="invalid"


def test_invalid_and_conflicting_model_passes_fail_closed():
    p=build_review_packet(candidate())
    a=verify_review_response(supported(),p)
    invalid=verify_review_response('{"verdict":"supported"}',p)
    bad=consensus(p,[a,invalid])
    assert bad["conclusion"]=="invalid_reviewer_output"
    assert bad["authorized_train_label_count"]==0
    b=verify_review_response(negative("unsupported"),p)
    assert consensus(p,[a,b])["conclusion"]=="flagged_unsupported"


def test_local_only_reviewer_and_positive_only_quote_gate():
    with pytest.raises(ValueError,match="loopback"):
        LocalSemanticReviewer("https://api.example.com/v1/chat/completions","remote")
    with pytest.raises(ValueError):
        LocalSemanticReviewer("http://0.0.0.0:18986/v1/chat/completions","unsafe")
    c=candidate()
    c["verdict"]="quote_not_exact_tool_observation"
    with pytest.raises(ValueError,match="quote-audited"):
        build_review_packet(c)
    c=candidate()
    c["evidence"][1]["doc_id"]="doc-a"
    with pytest.raises(ValueError,match="distinct"):
        build_review_packet(c)


def test_review_packet_rejects_extra_untrusted_fields():
    c=candidate()
    c["gold_answer"]="injection"
    with pytest.raises(ValueError,match="only independently"):
        build_review_packet(c)
