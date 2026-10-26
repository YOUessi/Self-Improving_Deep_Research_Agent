"""Gold-blind V2 lexical answer choice regression tests."""
import inspect

import pytest
from sidra.candidate_proposer import candidate_spans_v2,question_compare_options


@pytest.mark.parametrize("question,evidence,expected",[
    (
        "The meeting was chaired by what Austrian statesman?",
        "The Congress of Vienna was chaired by Austrian statesman "
        "Klemens von Metternich, and was held from 1814 to 1815.",
        "Klemens von Metternich",
    ),
    (
        "When was the director of Just Married born?",
        "Shawn Levy (born July 23, 1968) is a Canadian film director "
        "who directed the films Big Fat Liar and Just Married.",
        "July 23, 1968",
    ),
    (
        "How often is the journal released?",
        "The School Library Journal is a monthly magazine with articles "
        "and reviews for school librarians.",
        "monthly",
    ),
    (
        "How many people died in the riot?",
        "The Astor Place Riot, on May 10, 1849, left at least 25 dead "
        "and more than 120 injured.",
        "at least 25",
    ),
    (
        "Where are the headquarters of the company?",
        "Lenovo is a technology company with headquarters in "
        "Beijing, China and Morrisville, North Carolina. "
        "It sells personal computers.",
        "Beijing, China and Morrisville, North Carolina",
    ),
    (
        "Which is taller, 80 South Street or Central Park Place?",
        "80 South Street was proposed in 2016 in New York City, "
        "and Central Park Place was built in 1988.",
        "80 South Street",
    ),
    (
        "What video game was run by the engine?",
        "The MacVenture engine first ran Déjà Vu: A Nightmare Comes True, "
        "followed by Uninvited and Shadowgate.",
        "Déjà Vu",
    ),
])
def test_expected_spans_from_visible_text_only(question,evidence,expected):
    output=candidate_spans_v2(evidence,question,limit=8)
    assert expected in output,(expected,output)


def test_comparison_alternatives_from_question_are_not_hidden_gold():
    q="Which is taller, 80 South Street or Central Park Place?"
    assert question_compare_options(q)==["80 South Street","Central Park Place"]
    assert question_compare_options("What was the founding year?")==[]


def test_proposer_does_not_accept_question_or_evaluator_gold_objects():
    param_names=inspect.signature(candidate_spans_v2).parameters
    assert set(param_names)=={"text","question","limit"}
    assert not any("gold" in k or "support" in k or "label" in k
                   for k in param_names)
    with pytest.raises(TypeError):
        candidate_spans_v2("evidence","question",gold_answer="PRIVATE")
    with pytest.raises(ValueError,match="limit"):
        candidate_spans_v2("evidence","question",limit=0)


def test_bounded_deterministic_and_empty_input():
    evidence=("Mason Bridge is in Riverhaven, and it was built by "
              "Elizabeth von Hohenberg in July 23, 1968.")
    question="Who built Mason Bridge?"
    r1=candidate_spans_v2(evidence,question,limit=3)
    r2=candidate_spans_v2(evidence,question,limit=3)
    assert r1==r2 and 0<len(r1)<=3
    assert candidate_spans_v2("  ",question)==[]
