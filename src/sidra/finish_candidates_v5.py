"""V5 answer options derived ONLY from actually READ source text.

V4 has a hard four-answer catalog ceiling and ranks only the last couple
of short extract lists. V5 interleaves *both* observed sources, expands the
extractor budget when only a finish/abstain action remains, and makes 'yes'
and 'no' legal for genuine polar questions. It does not know which option
is correct. The on-policy categorical learner must still choose it.

V5 must be evaluated as a NEW legal-action policy class: identical old/new
model comparisons require matched v5 catalogs, normalizers and budget.
"""
from __future__ import annotations

import re
from sidra.candidate_proposer import candidate_spans_v2
from sidra.metrics import normalize_answer


_POLAR=re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|should|would|"
    r"will|has|have|had|may|might|must)\b",
    re.IGNORECASE,
)


def is_polar_question(question:str)->bool:
    """Structural question-form detection, not gold-answer prediction."""
    return bool(_POLAR.match(question))


def visible_finish_options(question:str,read_observations:list[dict], *,
                           limit:int=8)->list[str]:
    if not 1<=limit<=20:
        raise ValueError("finish candidate cap must be 1..20")
    if not question.strip() or not read_observations:
        return []
    reads=[o for o in read_observations if
           o.get("kind")=="read" and o.get("text")]
    if not reads:return []
    # Only text already returned by READ; never use corpus-wide titles,
    # gold answer, supporting-doc labels or DEV evaluator outputs.
    spans=[]
    observed=reads[-2:]
    for o in reversed(observed):
        spans.append(candidate_spans_v2(
            str(o["text"])[:3500],question,limit=max(16,limit*3)))
    choices=[]
    if is_polar_question(question):
        # Make BOTH options legal. Never inject the gold Boolean direction.
        choices.extend(["yes","no"])
    for index in range(max(map(len,spans),default=0)):
        for source_spans in spans:
            if index<len(source_spans):
                choices.append(source_spans[index])
    result=[]
    seen=set()
    for answer in choices:
        norm=normalize_answer(answer)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(answer)
        if len(result)>=limit:break
    return result
