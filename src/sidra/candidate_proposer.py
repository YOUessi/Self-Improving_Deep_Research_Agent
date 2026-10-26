"""Non-oracle lexical answer candidates proposed ONLY from tool-visible text.

No Question object, no source/gold labels, no corpus-wide documents, no dev.
This intentionally does NOT infer correctness or factual entailment. It provides
a bounded answer-choice catalog for exact-normalized on-policy RL; success still
requires the LM to select the correct candidate and read/cite real evidence.

V2 focuses on Unicode person names, multiword names with "von/de/and",
complete calendar dates, quantitative qualifiers, frequency, cities and
question-listed comparison choices. It ranks using surface question/evidence
overlap, never hidden answer labels.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_CAPITAL = r"[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’\-]*"
_LINK = r"(?:of|the|de|da|del|du|la|le|van|von|der|den|and|&)"
_PROPER = re.compile(
    rf"\b{_CAPITAL}(?:\s+(?:{_CAPITAL}|{_LINK})){{0,5}}\b")
_MONTH = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)
_DATE = re.compile(
    rf"\b(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+"
    r"(?:1[5-9]|20)\d{2}\b",re.I)
_YEAR = re.compile(r"\b(?:1[5-9]|20)\d{2}\b")
_QUANTITY = re.compile(
    r"\b(?:(?:at least|at most|more than|over|fewer than|"
    r"approximately|about|around|nearly|almost)\s+)?"
    r"\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s+(?:million|billion))?\b",
    re.I)
_FREQUENCY = re.compile(
    r"\b(?:daily|weekly|biweekly|fortnightly|monthly|"
    r"quarterly|annually|yearly|semiannually|twice a week|once a year)\b",
    re.I)
_HQ = re.compile(
    r"\bheadquarters?\s+(?:are|is|were|was|located|based)?\s*in\s+"
    r"([A-ZÀ-ÖØ-Þ][^.;\n]{4,115})",re.I)
_QUESTION_STOP={
    "what","where","when","which","who","whose","whom","how","that","this",
    "were","was","are","is","does","did","with","from","about","there",
    "whose","have","into","been","during","after","before","would","some",
    "name","the","and","for","of","their","located","country","city",
    "between","year","was","are",
}
_STOP_ENTITIES={
    "the","a","an","this","that","it","they","he","she","as","at","in",
    "on","its","after","before","when","who","what","where","which","during",
    "there","his","her","from","for","by","and","but","the first",
}


def _norm(s:str)->str:
    return re.sub(r"\s+"," ",s).strip(" \t\n.;:!?\"'“”‘’()[]")


def _wordset(text:str)->set[str]:
    return {w for w in re.findall(r"\b[\wÀ-ÿ]{3,}\b",text.casefold())
            if w not in _QUESTION_STOP}


def _sentences(text:str)->list[str]:
    # Split only at actual sentence endings; preserve comma-delimited compound
    # dates, place names and coordinated person names.
    return [p for p in re.split(r"(?<=[.!?])\s+|\n+",text[:3500])
            if p.strip()]


def _add(candidates:dict,token:str,score:float,context:str, *,
         kind:str="name")->None:
    token=_norm(token)
    if not 2<=len(token)<=135 or token.casefold() in _STOP_ENTITIES:
        return
    if not re.search(r"[\wÀ-ÿ]",token):
        return
    key=token.casefold()
    previous=candidates.get(key)
    if previous is None or score>previous[0]:
        candidates[key]=(score,token,context,kind)


def question_compare_options(question:str)->list[str]:
    """Choice questions may legitimately be answered with a noun from question."""
    q=" ".join(question.split())
    m=re.search(
        r"\bwhich\s+(?:is|was|were|are)\s+"
        r"(?:taller|older|younger|larger|smaller|longer|higher|first)"
        r"\s*,?\s*(.+?)\s+or\s+(.+?)(?:\?|$)",
        q,re.I)
    if not m:return []
    answers=[_norm(m.group(1)),_norm(m.group(2))]
    return [x for x in answers if 2<=len(x)<=90]


def candidate_spans_v2(text:str,question:str,limit:int=3)->list[str]:
    """Rank lexical answer spans solely from visible passage + question."""
    if limit<1:raise ValueError("positive candidate limit required")
    if not text.strip() or not question.strip():return []
    qlow=question.casefold()
    when=bool(re.search(r"\bwhen\b|\bwhat\s+year\b",qlow))
    how_many=bool(re.search(r"\bhow\s+(?:many|much)\b",qlow))
    how_often="how often" in qlow
    where=("where" in qlow or "headquarters" in qlow)
    what_region="region" in qlow
    who=bool(re.search(r"\bwho\b|\bwhom\b|\bwhat\s+.*statesman",qlow))
    comparison=bool(question_compare_options(question))
    qterms=_wordset(question)
    by_name={}
    previous={}
    sentences=_sentences(text)
    options=question_compare_options(question)
    for index,sentence in enumerate(sentences):
        sent_terms=_wordset(sentence)
        overlap=len(sent_terms&qterms)
        sent_score=math.log1p(overlap)*1.3
        # Longer passages and source order must not drown out question clues.
        bonus=max(0,0.45-0.015*index)
        refs=[]

        for m in _PROPER.finditer(sentence):
            phrase=_norm(m.group(0))
            words=phrase.split()
            if not phrase:continue
            refs.append((phrase,"name"))
            # Coordinated person names: "Jim Bob and Michelle Duggar".
            if len(words)>=4:
                refs.append((" ".join(words[:-1]),"name_prefix"))
            # Honor short trailing proper names when full phrase contains the
            # publisher/organization prefix.
            if len(words)>=3:
                refs.append((" ".join(words[-2:]),"name_suffix"))
            if len(words)>=4:
                refs.append((" ".join(words[:3]),"name_prefix"))
        for m in _DATE.finditer(sentence):
            refs.append((_norm(m.group(0)),"date"))
        for m in _YEAR.finditer(sentence):
            refs.append((m.group(0),"year"))
        for m in _QUANTITY.finditer(sentence):
            refs.append((_norm(m.group(0)),"quantity"))
        for m in _FREQUENCY.finditer(sentence):
            refs.append((_norm(m.group(0)),"frequency"))
        for m in _HQ.finditer(sentence):
            refs.append((_norm(m.group(1)),"headquarters"))
        for candidate in options:
            if candidate.casefold() in sentence.casefold():
                refs.append((candidate,"comparison"))
        for raw,kind in refs:
            if not raw:continue
            idx=sentence.casefold().find(raw.casefold())
            local=sentence[max(0,idx-70):idx+len(raw)+70].casefold()
            score=1.1+sent_score+bonus
            if raw.casefold() in qlow and kind!="comparison":
                score-=2.5
            if when:
                score+=(8 if kind=="date" else 4 if kind=="year" else -1.5)
                if "born" in qlow and "born" in local:score+=2.3
            if how_many:
                score+=(6.5 if kind=="quantity" else -2.5)
            if how_often:
                score+=(12 if kind=="frequency" else -2.5)
            if where:
                score+=(9 if kind=="headquarters" else 1.0 if kind=="name" else -1)
            if what_region:
                score+=7 if "region" in raw.casefold() else -1
            if comparison:
                score+=9 if kind=="comparison" else -3
            if who:
                score+=2.2 if kind.startswith("name") else -1.5
            if "company" in qlow or "subsidiary" in qlow:
                if kind.startswith("name") and any(
                    x in local for x in ("company","subsidiary","owned")):
                    score+=2.5
            if "team" in qlow or "won" in qlow:
                if kind.startswith("name") and any(
                    x in local for x in ("won","win","beat","champion")):
                    score+=2
            if "statesman" in qlow and "statesman" in local:
                score+=3.4
            if kind.startswith("name") and len(raw.split())>=2:
                score+=0.5
            if kind=="name_prefix":score-=0.8
            if kind=="name_suffix":score-=0.7
            _add(by_name,raw,score,sentence,kind=kind)
            previous[raw.casefold()]=previous.get(raw.casefold(),0)+1

    # Mention frequency is weak lexical evidence only, never gold supervision.
    rescored=[]
    for key,(s,raw,context,kind) in by_name.items():
        freq_bonus=min(3,previous.get(key,1)-1)*0.35
        rescored.append((-(s+freq_bonus),raw.casefold(),raw,kind))
    rescored.sort()
    return [item[2] for item in rescored[:limit]]
