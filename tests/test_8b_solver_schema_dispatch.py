"""Regression: the HTTP response parser must accept the actual schema per mode.

Prior quote/sentence-ID pilots returned HTTP 200 and valid JSON but were all
reported failed: they incorrectly reused the old 'answer+citations' validator.
This test protects explicit field-level protocol differentiation, without
requiring a running model endpoint.
"""
import json

import pytest

from scripts.m7_fixed_evidence_8b_solver_pilot import get_solver_output


class FakeHTTP:
    def __init__(self,payload):self.payload=json.dumps(payload).encode()
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,size=-1):return self.payload


def fake_open_for(monkeypatch,response):
    body={
        "choices":[{"message":{"content":json.dumps(response)},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":30,"completion_tokens":18},
    }
    monkeypatch.setattr(
        "scripts.m7_fixed_evidence_8b_solver_pilot.urlopen",
        lambda req,timeout:FakeHTTP(body),
    )


def test_three_explicit_protocols_are_accepted_with_correct_contract(monkeypatch):
    for response,expected in [
        ({"answer":"Port","citations":["a"]},frozenset({"answer","citations"})),
        ({"answer":"Port","evidence":[{"doc_id":"a","quote":"A sufficiently long quote"}]},
         frozenset({"answer","evidence"})),
        ({"answer":"Port","sentence_ids":["d0s1"]},
         frozenset({"answer","sentence_ids"})),
    ]:
        fake_open_for(monkeypatch,response)
        parsed,usage,elapsed=get_solver_output(
            "http://127.0.0.1:18985/v1/chat/completions",
            {"model":"fake","messages":[]},expected_keys=expected)
        assert parsed==response
        assert usage["completion_tokens"]==18
        assert elapsed>=0


def test_wrong_contract_rejected_instead_of_silent_skip(monkeypatch):
    response={"answer":"Port","sentence_ids":["d0s1"]}
    fake_open_for(monkeypatch,response)
    with pytest.raises(ValueError,match="Unexpected solver schema"):
        get_solver_output("http://127.0.0.1:18985/v1/chat/completions",
                          {"model":"fake","messages":[]})
