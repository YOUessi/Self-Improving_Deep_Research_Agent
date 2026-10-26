"""TRAIN-only *model-driven* RSI Challenger candidate generation.

Receives ONLY CHALLENGER_INPUT.jsonl (opaque id, original TRAIN question).
It NEVER reads verifier maps, gold answers, DEV/eval-A/final-B, checkpoints
or raw heldout items. Generated novel questions are UNLABELLED until a
separate, independently grounded TRAIN-only Verifier certifies them.

Backend: local OpenAI-compatible Qwen3 inference on 127.0.0.1 only.
This is not a self-training algorithm or reward model.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from sidra.rsi_challenge_scheduler import _norm


CHALLENGER_SYSTEM = (
    "You are a Challenger task generator for multi-hop fact verification. "
    "You receive one ordinary TRAIN question, WITHOUT the answer or any "
    "supporting-document labels. Produce ONE different, precise, factual "
    "multi-hop question that an agent might answer by searching and reading "
    "trusted external text. Do not answer either question. Do not invent "
    "factually unsupported assertions, ask for private data, or reference "
    "the hidden answer. Output only a JSON object with exactly two string "
    "fields: question (a NEW question) and reason (short description of "
    "what additional verification is required)."
)


@dataclass(frozen=True)
class ChallengerOutcome:
    challenge_id: str
    proposal_question: str
    reason: str
    verdict: str
    model_completion_sha256: str

    def as_dict(self) -> dict:
        return {
            "challenge_id":self.challenge_id,
            "proposal_question":self.proposal_question,
            "reason":self.reason,
            "verdict":self.verdict,
            "model_completion_sha256":self.model_completion_sha256,
        }


def challenger_messages(source_question:str)->list[dict]:
    if not source_question.strip():
        raise ValueError("empty TRAIN question")
    return [
        {"role":"system","content":CHALLENGER_SYSTEM},
        {"role":"user","content":json.dumps({
            "original_train_question":source_question,
            "instruction":"Create a separate harder research question. "
                          "Do not provide an answer or assumed facts.",
        },ensure_ascii=False)},
    ]


def parse_challenge(
    raw:str,*,challenge_id:str,source_question:str,
    prior_proposals:list[str]|None=None)->ChallengerOutcome:
    fingerprint=hashlib.sha256(raw.encode("utf-8")).hexdigest()
    def reject(code:str)->ChallengerOutcome:
        return ChallengerOutcome(challenge_id,"","",code,fingerprint)
    # Qwen3 non-thinking structured output should be strict JSON; do not
    # treat model internal reasoning or arbitrary tool commands as tasks.
    try:
        decoded=json.loads(raw)
    except (TypeError,json.JSONDecodeError):
        return reject("malformed_json")
    if not isinstance(decoded,dict) or set(decoded)!={"question","reason"}:
        return reject("invalid_schema")
    question,reason=decoded["question"],decoded["reason"]
    if not isinstance(question,str) or not isinstance(reason,str):
        return reject("invalid_types")
    question=question.strip()
    reason=reason.strip()
    if not 25<=len(question)<=450 or not question.endswith("?"):
        return reject("not_a_bounded_factual_question")
    if not 8<=len(reason)<=350:
        return reject("missing_short_justification")
    # These strings are schema/role names and must never be treated as
    # model-supplied "proof" of a training label.
    banned=("gold_answer","supporting_doc_ids","test_label","dev_label",
            "the hidden answer is","the correct answer is")
    if any(x in question.casefold() or x in reason.casefold() for x in banned):
        return reject("references_hidden_labels")
    if (_norm(question)==_norm(source_question) or
        SequenceMatcher(None,_norm(question),_norm(source_question)).ratio()>0.98):
        return reject("repeats_original_question")
    if any(SequenceMatcher(
           None,_norm(question),_norm(previous)).ratio()>=0.92
           for previous in (prior_proposals or [])):
        return reject("near_duplicate_previous_candidate")
    return ChallengerOutcome(
        challenge_id,question,reason,
        "unlabelled_candidate_pending_independent_verification",fingerprint)


class LocalOpenAIChallenger:
    """Local-only inference; credentials and private data never transmitted."""

    def __init__(self,endpoint:str,model:str,*,temperature:float=0.65,
                 top_p:float=0.95,max_tokens:int=180,timeout_seconds:int=90):
        parsed=urlparse(endpoint)
        if (parsed.scheme!="http" or
            parsed.hostname not in ("localhost","127.0.0.1") or
            parsed.path!="/v1/chat/completions" or
            parsed.username is not None or parsed.password is not None):
            raise ValueError("Challenger only permits local HTTP endpoint")
        if not model or not 0<=temperature<=1 or not 0<top_p<=1:
            raise ValueError("invalid model/sampling settings")
        self.endpoint=endpoint
        self.model=model
        self.temperature=temperature
        self.top_p=top_p
        self.max_tokens=max_tokens
        self.timeout_seconds=timeout_seconds

    def generate(self,source_question:str,*,seed:int)->str:
        payload={
            "model":self.model,
            "messages":challenger_messages(source_question),
            "temperature":self.temperature,
            "top_p":self.top_p,
            "seed":seed,
            "max_tokens":self.max_tokens,
            "response_format":{"type":"json_object"},
            "extra_body":{"chat_template_kwargs":{"enable_thinking":False}},
        }
        request=Request(
            self.endpoint,
            data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={"Content-Type":"application/json"},method="POST")
        with urlopen(request,timeout=self.timeout_seconds) as reply:
            response=json.loads(reply.read(400_000).decode())
        content=response["choices"][0]["message"]["content"]
        if not isinstance(content,str):
            raise ValueError("model returned no text candidate")
        return content


def generate_train_candidates(
    queue_file:Path,manifest_file:Path,out:Path,*,backend:LocalOpenAIChallenger,
    seed:int,limit:int=0)->dict:
    if out.exists():raise FileExistsError(out)
    manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
    digest=hashlib.sha256(queue_file.read_bytes()).hexdigest()
    if digest!=manifest.get("challenger_input_sha256") or manifest.get("split")!="train":
        raise ValueError("Challenger input checksum or TRAIN provenance invalid")
    if manifest.get("model_driven_challenger_executed"):
        raise ValueError("input cannot contain recursive proposed data")
    rows=[json.loads(x) for x in queue_file.read_text(encoding="utf-8").splitlines()
          if x.strip()]
    if len(rows)!=manifest["n"]:
        raise ValueError("Challenger input question count mismatch")
    if not rows or len({x.get("challenge_id") for x in rows})!=len(rows):
        raise ValueError("empty/duplicate Challenger IDs")
    if any(set(x)!={"challenge_id","question"} for x in rows):
        raise ValueError("Challenger data contains leaked extra fields")
    chosen=rows[:limit] if limit>0 else rows
    candidates=[]
    previous=[]
    for i,row in enumerate(chosen):
        try:
            raw=backend.generate(row["question"],seed=seed+i)
            outcome=parse_challenge(
                raw,challenge_id=row["challenge_id"],
                source_question=row["question"],prior_proposals=previous)
        except (URLError,ValueError,KeyError,IndexError,TimeoutError,OSError):
            outcome=ChallengerOutcome(
                challenge_id=row["challenge_id"],
                proposal_question="",reason="",verdict="inference_error",
                model_completion_sha256="")
        candidates.append(outcome.as_dict())
        if outcome.verdict=="unlabelled_candidate_pending_independent_verification":
            previous.append(outcome.proposal_question)
    out.mkdir(parents=True,exist_ok=False)
    path=out/"UNLABELLED_CHALLENGER_CANDIDATES.jsonl"
    with path.open("x",encoding="utf-8") as stream:
        for row in candidates:
            stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    counts=Counter(r["verdict"] for r in candidates)
    summary={
        "status":"challenger_candidate_generation_not_RSI_training",
        "model":backend.model,
        "backend_local_only":True,
        "generation_seed":seed,
        "source_queue_sha256":digest,
        "source_queue_generation":manifest["generation"],
        "source_split":"train",
        "n_attempted":len(chosen),
        "n_structurally_valid_novel_candidates":counts.get(
            "unlabelled_candidate_pending_independent_verification",0),
        "verdict_histogram":dict(counts),
        "challenger_gold_label_access":False,
        "independent_verifier_approved_novel_tasks":0,
        "trainable_new_gold_labels_produced":0,
        "optimizer_updates":0,
        "source_document_semantic_entailment_verified":False,
        "original_model_completion_content_logged":False,
        "critical_caveat":"Candidate questions are unlabelled and must NOT "
             "inherit original TRAIN answers. This experiment measures raw "
             "task-generation validity only, not RSI learning.",
    }
    (out/"CHALLENGER_SUMMARY.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary
