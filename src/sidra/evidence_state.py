"""Evidence-state representation shared by Search/Read/Finish and TRAIN rewards.

This is a DETERMINISTIC observable evidence ledger, not an oracle or a
semantic entailment engine. At inference it takes only question, actual
SEARCH hit metadata and actual READ text. It has no access to Question.gold,
supporting_doc_ids, unreturned corpus text, or evaluator classifications.

One ledger instance is recomputed from the *real observation history* at
each decision. The same ledger can be reconstructed post-rollout for
TRAIN-only shaping/credit assignment. State summaries are safe to expose to
the one-token finite Agent; gold labels remain isolated in TRAIN scorers.

Proxy features are NOT claims of factual entailment. Cross-document title
links can be wrong, and lexical coverage is only a research heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from sidra.bridge import bridge_candidates
from sidra.retrieval import tokenize
from sidra.schema import Action


_STOP=frozenset({
    "this","that","these","those","they","their","them","who","which",
    "when","where","what","whether","how","why","were","was","are","is",
    "with","from","the","and","for","by","any","both","into","also",
    "known","called","name","whose","did","does","can","would","has",
    "had","have","there","about","after","before","first","second",
    "one","two","what","city","place","person","country","years",
})
_POLAR=frozenset({"yes","no"})


def _terms(s:str)->set[str]:
    return {t for t in tokenize(s) if len(t)>=2 and t not in _STOP}


def _normalized(s:str)->str:
    return " ".join(tokenize(s))


def _phrase_in(phrase:str,text:str)->bool:
    p=_normalized(phrase)
    h=_normalized(text)
    return bool(p and f" {p} " in f" {h} ")


def _question_sentence_overlap(question_terms:set[str],text:str)->float:
    if not question_terms:
        return 0.0
    import re
    pieces=re.split(r"(?<=[.!?])\s+|\n+",text[:3500])
    values=[len(_terms(piece)&question_terms)/len(question_terms)
            for piece in pieces[:80]]
    return max(values,default=0.0)


@dataclass(frozen=True)
class EvidenceRead:
    doc_id:str
    title:str
    text:str
    relevance:float


@dataclass(frozen=True)
class EvidenceState:
    question:str
    searched_queries:tuple[str,...]
    exposed_hits:tuple[tuple[str,str,str],...]  # (id,title,snippet)
    read_sources:tuple[EvidenceRead,...]
    linked_pairs:tuple[tuple[str,str],...]
    anchor_coverage:float
    next_bridge_query:str | None
    potential:float

    @property
    def read_ids(self)->frozenset[str]:
        return frozenset(d.doc_id for d in self.read_sources)

    @property
    def available_unread(self)->tuple[str,...]:
        return tuple(did for did,_,_ in self.exposed_hits
                     if did not in self.read_ids)

    @property
    def evidence_gap(self)->str:
        if not self.exposed_hits:return "need_search"
        if not self.read_sources:return "need_first_read"
        if len(self.read_sources)==1:
            return "need_second_source" if self.available_unread else "need_second_search"
        if not self.linked_pairs:
            return "unlinked_sources_need_verification"
        return "linked_sources_answer_check"

    def public_summary(self)->dict:
        """No labels, source text, private annotations or score of correctness."""
        return {
            "phase":self.evidence_gap,
            "distinct_read_sources":len(self.read_sources),
            "unread_search_hits":len(self.available_unread),
            "cross_document_title_links":len(self.linked_pairs),
            "question_anchor_coverage":round(self.anchor_coverage,4),
            "next_visible_bridge_query":self.next_bridge_query,
            "observable_progress_proxy":round(self.potential,5),
            "warning":"lexical/graph evidence proxy; NOT proven entailment",
        }

    def candidate_support(self,answer:str)->tuple[int,float]:
        """Count source-local literal answer hits and question-sentence overlap."""
        answer=answer.strip()
        if not answer or answer.casefold() in _POLAR:return 0,0.0
        matched=[d for d in self.read_sources if _phrase_in(answer,d.text)]
        if not matched:return 0,0.
        qterms=_terms(self.question)
        value=max((_question_sentence_overlap(
            qterms,sentence) for d in matched
            for sentence in _sentences_with_answer(d.text,answer)),default=0.)
        return len(matched),round(value,6)

    def action_biases(self,actions:Sequence[Action])->list[float]:
        """Gold-blind state-conditioned logits for *all* allowed action kinds.

        Additive nonoracle logits belong to both behavior sampling and
        differentiable re-scoring. No action is granted an invalid DocID.
        Biases are capped; uncertainty never silently forces a wrong answer.
        """
        qterms=_terms(self.question)
        reads=self.read_sources
        nread=len(reads)
        seen={did:(title,snippet) for did,title,snippet in self.exposed_hits}
        scores=[]
        for a in actions:
            score=0.
            if a.kind=="read":
                if a.doc_id not in seen or a.doc_id in self.read_ids:
                    raise ValueError("READ action refers to an unseen/already read document")
                title,snippet=seen[a.doc_id]
                overlap=(len(qterms&(_terms(title)|_terms(snippet)))/
                         max(1,len(qterms)))
                bridge_link=any(_phrase_in(title,d.text) for d in reads)
                score=(0.65 if nread==0 else 0.35) + 0.8*overlap
                score+=1.0 if bridge_link else 0.0
                # After two reads, a third irrelevant read is less valuable.
                if nread>=2 and not bridge_link and overlap<0.20:
                    score-=0.4
            elif a.kind=="search":
                if _normalized(a.query) in {_normalized(x) for x in self.searched_queries}:
                    raise ValueError("Search policy attempted a repeated query")
                if not reads:
                    score=0.3
                else:
                    bridge=(bool(self.next_bridge_query)
                            and _normalized(a.query)==_normalized(self.next_bridge_query))
                    score=(1.15 if bridge and not self.available_unread else
                           0.5 if bridge else 0.1)
                    if self.evidence_gap=="linked_sources_answer_check":
                        score-=0.45
            elif a.kind=="finish":
                if not a.answer:
                    # Abstention stays possible, especially if the budget
                    # has been exhausted; never treat no evidence as proof.
                    score=-0.45 if nread<2 else -0.15
                else:
                    hits,cooccurrence=self.candidate_support(a.answer)
                    if nread<2 or not set(a.citations)<=self.read_ids:
                        raise ValueError("nonempty answer violates READ evidence precondition")
                    score=(0.95 if hits else -1.0)+0.55*cooccurrence
                    if self.linked_pairs:
                        score+=0.35
                    # No literal yes/no answer need be present in paragraph,
                    # but polarity still requires at least two READ sources.
                    if a.answer.casefold() in _POLAR:
                        score=0.1+0.3*int(bool(self.linked_pairs))
            else:
                raise ValueError("unknown legal evidence action")
            scores.append(round(max(-2.0,min(2.0,score)),6))
        return scores


def _sentences_with_answer(text:str,answer:str)->list[str]:
    import re
    parts=re.split(r"(?<=[.!?])\s+|\n+",text[:3500])
    return [x for x in parts[:80] if _phrase_in(answer,x)]


def derive_evidence_state(question:str,observations:list[dict])->EvidenceState:
    if not question.strip():
        raise ValueError("nonempty question required")
    queries:list[str]=[]
    hits:dict[str,tuple[str,str]]={}
    reads:dict[str,EvidenceRead]={}
    qterms=_terms(question)
    for obs in observations:
        if not isinstance(obs,dict):raise TypeError("invalid tool observation")
        kind=obs.get("kind")
        if kind=="search":
            query=str(obs.get("query","")).strip()
            if query:queries.append(query)
            for hit in obs.get("hits",[]):
                if not isinstance(hit,dict):continue
                did=hit.get("doc_id")
                if not isinstance(did,str) or not did:continue
                hits.setdefault(did,(str(hit.get("title",""))[:160],
                                     str(hit.get("snippet",""))[:500]))
        elif kind=="read":
            did=obs.get("doc_id")
            if not isinstance(did,str) or not did:continue
            # Only actual READ results, not search-hit snippets.
            text=str(obs.get("text",""))[:3500]
            if not text:continue
            title=str(obs.get("title",""))[:160]
            frac=_question_sentence_overlap(qterms,text)
            reads.setdefault(did,EvidenceRead(did,title,text,round(frac,6)))
    source_list=tuple(reads.values())
    pairs=[]
    for i,a in enumerate(source_list):
        for b in source_list[i+1:]:
            if (_phrase_in(a.title,b.text) and len(_terms(a.title))>=1 or
                _phrase_in(b.title,a.text) and len(_terms(b.title))>=1):
                pairs.append(tuple(sorted((a.doc_id,b.doc_id))))
    anchors=set()
    for source in source_list:
        # Limit to question-relevant document spans: not whole documents,
        # where unrelated text can trivially match all question tokens.
        anchors.update(_terms(source.title))
        anchors.update(_terms(source.text[:500]))
    anchor_coverage=len(qterms&anchors)/max(1,len(qterms))
    gap=None
    if source_list:
        last=source_list[-1]
        for proposed in bridge_candidates(
            question,last.title,last.text[:2800],
            prior_queries=set(queries))[:8]:
            # Do not search again for a document title already exposed; READ
            # it instead. No hidden second-hop title lookup.
            query=_normalized(proposed.query)
            if query and query not in {_normalized(x) for x in queries}:
                if not any(query==_normalized(t) for t,_ in hits.values()):
                    gap=proposed.query
                    break
    frontier=tuple((did,*data) for did,data in hits.items())
    # A BOUNDED progress proxy; increasing irrelevant READ count is weakly
    # rewarded, but true evidence/answer correctness is never assumed.
    potential=(0.10*int(bool(hits))
               +0.25*min(len(source_list),2)/2
               +0.30*anchor_coverage
               +0.25*int(bool(pairs))
               +0.10*int(bool(source_list and not gap)))
    potential=max(0.,min(1.,potential))
    return EvidenceState(
        question=question,searched_queries=tuple(queries),
        exposed_hits=frontier,read_sources=source_list,
        linked_pairs=tuple(sorted(set(pairs))),
        anchor_coverage=round(anchor_coverage,6),
        next_bridge_query=gap,
        potential=round(potential,6),
    )
