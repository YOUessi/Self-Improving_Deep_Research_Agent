"""Grounded RSI Challenger: strict TRAIN-only retrieval packet, quote audit.

This creates *candidate* new questions using ordinary BM25 search/read text,
not gold/supporting_fact labels. The model proposes an answer and at least
two exact source quotes; an independent deterministic Verifier checks those
quotes against the original corpus. It does NOT prove semantic entailment.

CRITICAL: Even when the quote gate passes, NEW questions are UNLABELLED and
ineligible for training until a separate evidence/semantic verifier approves
their ground-truth answer. Generated assertions can contain false premises.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict,dataclass
from difflib import SequenceMatcher
from urllib.request import Request,urlopen

from sidra.retrieval import BM25Index
from sidra.rsi_model_challenger import LocalOpenAIChallenger


SYSTEM=(
    "You are a research question Challenger operating ONLY on the supplied "
    "TRUSTED document excerpts. Your goal is one NEW, answerable, precise "
    "multi-hop question. Never invent people, dates, role titles, relationships "
    "or premises absent from the excerpts. If the snippets cannot support "
    "two different document references and an extractive answer, output "
    "a JSON object with question, proposed_answer, reason and evidence set "
    "to empty strings/list. Otherwise output exactly this JSON object: "
    "{question:string, proposed_answer:string, reason:string, "
    "evidence:[{doc_id:string,quote:string},{doc_id:string,quote:string}]}. "
    "Each quote must be copied VERBATIM from its document excerpt, contain "
    "the factual premise it is supposed to support, and be at least 20 "
    "characters long. The answer text must appear EXACTLY in one quote. "
    "Do not refer to labels, hidden answers, or any knowledge outside excerpts."
)


@dataclass(frozen=True)
class EvidenceExcerpt:
    doc_id:str
    title:str
    snippet:str


@dataclass(frozen=True)
class GroundingPacket:
    challenge_id:str
    source_question:str
    snippets:tuple[EvidenceExcerpt,...]
    packet_sha256:str

    def model_view(self)->dict:
        return {
            "challenge_id":self.challenge_id,
            "source_question":self.source_question,
            "search_results":[asdict(x) for x in self.snippets],
            "instruction":"Ask a NEW grounded multi-hop question with two "
                          "different supporting document excerpts.",
        }


def make_grounding_packet(
    challenge_id:str,source_question:str,*,scope:str,index:BM25Index,
    top_k:int=4,max_snippet_chars:int=550,
)->GroundingPacket:
    if not challenge_id.startswith("c-") or not source_question.strip():
        raise ValueError("invalid frozen Challenger identity")
    if not 2<=top_k<=8 or not 150<=max_snippet_chars<=1200:
        raise ValueError("invalid model exposure budget")
    hits=index.search(source_question,k=top_k,scope=scope)
    if len({x.doc_id for x in hits})!=len(hits):
        raise ValueError("duplicated BM25 evidence identity")
    snippets=tuple(EvidenceExcerpt(h.doc_id,h.title,h.snippet[:max_snippet_chars])
                   for h in hits)
    if len(snippets)<2:
        raise ValueError("insufficient reachable documents for grounded Challenger")
    serialized=json.dumps({
        "challenge_id":challenge_id,"source_question":source_question,
        "search_results":[asdict(x) for x in snippets],
    },sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return GroundingPacket(
        challenge_id,source_question,snippets,
        hashlib.sha256(serialized.encode()).hexdigest())


def grounded_messages(packet:GroundingPacket)->list[dict]:
    view=packet.model_view()
    if set(view)!={"challenge_id","source_question","search_results","instruction"}:
        raise ValueError("untrusted packet view")
    return [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":json.dumps(view,ensure_ascii=False)},
    ]


@dataclass(frozen=True)
class GroundedCandidate:
    challenge_id:str
    question:str
    proposed_answer:str
    reason:str
    evidence:tuple[dict,...]
    verdict:str
    packet_sha256:str
    completion_sha256:str

    def as_dict(self)->dict:
        out=asdict(self)
        out["evidence"]=list(self.evidence)
        return out


def verify_grounded_response(raw:str,packet:GroundingPacket)->GroundedCandidate:
    """Only structural/exact-quote verifier; NEVER grants a positive gold label."""
    fp=hashlib.sha256(raw.encode("utf-8")).hexdigest()
    def reject(code:str)->GroundedCandidate:
        return GroundedCandidate(packet.challenge_id,"","","",(),code,
                                 packet.packet_sha256,fp)
    try:x=json.loads(raw)
    except (ValueError,TypeError):return reject("malformed_json")
    if not isinstance(x,dict) or set(x)!={
        "question","proposed_answer","reason","evidence"}:
        return reject("invalid_schema")
    question=x["question"]
    answer=x["proposed_answer"]
    reason=x["reason"]
    evidence=x["evidence"]
    if (not all(isinstance(v,str) for v in (question,answer,reason)) or
        not isinstance(evidence,list)):
        return reject("invalid_field_types")
    question=question.strip()
    answer=answer.strip()
    reason=reason.strip()
    if not question or not answer or not evidence:
        return reject("model_abstained")
    if (not 25<=len(question)<=450 or not question.endswith("?") or
        not 1<=len(answer)<=120 or not 8<=len(reason)<=450):
        return reject("malformed_question_or_answer")
    if SequenceMatcher(None,question.casefold(),
                       packet.source_question.casefold()).ratio()>=0.98:
        return reject("question_not_novel")
    if len(evidence)!=2:
        return reject("requires_exactly_two_evidence_quotes")
    by_id={e.doc_id:e for e in packet.snippets}
    if len(by_id)!=len(packet.snippets):
        return reject("packet_duplicate_document")
    used=set()
    normalized_evidence=[]
    quoted_answer=False
    for citation in evidence:
        if (not isinstance(citation,dict)
            or set(citation)!={"doc_id","quote"}
            or not isinstance(citation["doc_id"],str)
            or not isinstance(citation["quote"],str)):
            return reject("invalid_evidence_schema")
        docid=citation["doc_id"]
        quote=citation["quote"]
        if docid not in by_id or docid in used:
            return reject("unretrieved_or_duplicate_citation")
        if not 20<=len(quote)<=max(len(by_id[docid].snippet),20):
            return reject("invalid_quote_length")
        if quote not in by_id[docid].snippet:
            return reject("quote_not_exact_tool_observation")
        quoted_answer|=(answer in quote)
        used.add(docid)
        normalized_evidence.append({"doc_id":docid,"quote":quote})
    if not quoted_answer:
        return reject("answer_not_extractive_in_provided_quotes")
    if any(word in (question+" "+reason).casefold() for word in (
        "gold_answer","supporting_doc_ids","hidden answer","private key","api key"
    )):
        return reject("references_forbidden_data")
    return GroundedCandidate(
        challenge_id=packet.challenge_id,question=question,
        proposed_answer=answer,reason=reason,
        evidence=tuple(normalized_evidence),
        verdict="quote_extracted_unlabelled_pending_semantic_verification",
        packet_sha256=packet.packet_sha256,completion_sha256=fp,
    )


class LocalGroundedChallenger(LocalOpenAIChallenger):
    """Inherit endpoint locality guard; send only exact gold-free packet."""

    def generate_packet(self,packet:GroundingPacket,*,seed:int)->str:
        payload={
            "model":self.model,
            "messages":grounded_messages(packet),
            "temperature":self.temperature,
            "top_p":self.top_p,
            "seed":seed,
            "max_tokens":self.max_tokens,
            "response_format":{"type":"json_object"},
            "chat_template_kwargs":{"enable_thinking":False},
        }
        request=Request(
            self.endpoint,
            data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={"Content-Type":"application/json"},method="POST")
        with urlopen(request,timeout=self.timeout_seconds) as reply:
            response=json.loads(reply.read(400_000).decode())
        text=response["choices"][0]["message"]["content"]
        if not isinstance(text,str):
            raise ValueError("no model-generated grounded candidate")
        return text
