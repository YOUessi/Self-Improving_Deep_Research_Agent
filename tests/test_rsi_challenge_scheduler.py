"""TRAIN-only, gold-redacted RSI Challenger data selection and strata."""
import hashlib,json

import pytest

from sidra.agent import AgentConfig
from sidra.rsi_challenge_scheduler import (
    stratify_train_questions,select_generation_challenges,freeze_challenger_queue,
)
from sidra.schema import Document,Question


def fixture():
    docs=[
        Document("a","Alpha Bridge","Alpha Bridge crosses the Lorn River"),
        Document("b","Beta Town","Beta Town is near Alpha Bridge"),
        Document("c","Gamma Valley","Gamma Valley contains an ancient river"),
        Document("d","Silent Hill","An unrelated historical location"),
    ]
    qs=[
        Question("t1","Which settlement neighbors Alpha Bridge and Beta Town?",
                 "Beta Town",("a","b"),split="train"),
        Question("t2","Alpha Bridge crosses which river in the region?",
                 "Lorn River",("a","d"),split="train"),
        Question("t3","What artist founded Zorbix Qantar?",
                 "Not in data",("c","d"),split="train"),
        Question("t4","Identify the two sites near Beta Town Alpha Bridge.",
                 "Beta Town",("a","b"),split="train"),
        Question("t5","Which bridge spans a local stream by Alpha Bridge?",
                 "Lorn River",("a","d"),split="train"),
        Question("t6","Who wrote the quasar magister pact?",
                 "Unknown",("c","d"),split="train"),
    ]
    return docs,qs


def test_strata_deterministic_and_gold_free_challenger(tmp_path):
    docs,qs=fixture()
    stratified=stratify_train_questions(
        qs,docs,AgentConfig(search_top_k=2))
    assert set(stratified)=={"direct","bridge","hard"}
    select=select_generation_challenges(
        stratified,seed=43,generation=1,desired=3)
    assert {x.difficulty_tier for x in select}=={"direct","bridge","hard"}
    again=select_generation_challenges(
        stratified,seed=43,generation=1,desired=3)
    assert [x.source_train_id for x in select]==[x.source_train_id for x in again]
    root=tmp_path/"frozen"
    manifest=freeze_challenger_queue(
        select,root,source_git_sha="a"*40,seed=43,generation=1)
    assert manifest["gold_fields_in_challenger_input"] is False
    assert manifest["model_driven_challenger_executed"] is False
    for line in (root/"CHALLENGER_INPUT.jsonl").read_text().splitlines():
        row=json.loads(line)
        assert set(row)=={"challenge_id","question"}
        assert "supporting_doc_ids" not in line and "gold_answer" not in line
        assert "source_train_question_id" not in line
    with pytest.raises(FileExistsError):
        freeze_challenger_queue(
            select,root,source_git_sha="a"*40,seed=43,generation=1)


def test_generations_do_not_reuse_train_ids_or_normalized_text(tmp_path):
    docs,qs=fixture()
    strat=stratify_train_questions(qs,docs,AgentConfig(search_top_k=2))
    first=select_generation_challenges(strat,seed=43,generation=1,desired=3)
    second=select_generation_challenges(
        strat,seed=43,generation=2,desired=3,
        prior_source_ids={x.source_train_id for x in first},
        prior_text_hashes={x.normalized_text_hash for x in first})
    assert not {x.source_train_id for x in first}&{
        x.source_train_id for x in second}
    assert not {x.normalized_text_hash for x in first}&{
        x.normalized_text_hash for x in second}


def test_fails_closed_on_dev_or_reserved_fingerprint():
    docs,qs=fixture()
    with pytest.raises(ValueError,match="TRAIN-only"):
        stratify_train_questions(
            [Question("dev1","Question for dev","answer",
                      ("a",),split="dev")],docs)
    with pytest.raises(ValueError,match="reserved frozen evaluation"):
        stratify_train_questions(qs,docs,forbidden_ids={"t1"})
    duplicate=hashlib.sha256(" ".join(qs[0].question.casefold().split()).encode()).hexdigest()
    with pytest.raises(ValueError,match="duplicates reserved eval"):
        stratify_train_questions(
            qs,docs,forbidden_normalized_hashes={duplicate})


def test_insufficient_unique_queue_does_not_fabricate_examples():
    docs,qs=fixture()
    strat=stratify_train_questions(qs,docs,AgentConfig(search_top_k=2))
    with pytest.raises(ValueError,match="insufficient"):
        select_generation_challenges(
            strat,seed=43,generation=1,desired=100)


def test_similarity_filter_does_not_promote_near_duplicate_question():
    docs,qs=fixture()
    similar=Question("t-near",
        qs[0].question.upper(),qs[0].gold_answer,
        supporting_doc_ids=qs[0].supporting_doc_ids,split="train")
    strat=stratify_train_questions(qs[:1]+[similar],docs,
                                    AgentConfig(search_top_k=2))
    with pytest.raises(ValueError,match="insufficient"):
        select_generation_challenges(strat,seed=43,generation=1,desired=2)
