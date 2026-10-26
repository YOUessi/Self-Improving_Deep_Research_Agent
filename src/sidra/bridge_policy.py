"""Gold-blind multi-hop policy that spends ONE additional SEARCH when useful.

Wrapper for the same underlying Qwen3 policy and GroundedPolicy validator.
- After first successful READ, propose a bridge query using only the
  QUESTION and actually read text, never gold labels or corpus-wide titles.
- Search through SearchAgent's usual audited tool, under the SAME max steps
  and max tool calls. It does not get an extra budget.
- After bridge SEARCH, choose an unread exposed hit with observed-title
  relevance. All other decisions delegated to the same GroundedPolicy.
- If there is not enough remaining budget to search, read and finish,
  do not force a bridge. Count all calls, including skipped model calls.
"""
from __future__ import annotations

from sidra.bridge import next_bridge_query, normalize
from sidra.schema import Action


def choose_unread_hit(question:str,observations:list[dict])->str|None:
    reads=[o for o in observations if o.get("kind")=="read"]
    searches=[o for o in observations if o.get("kind")=="search"]
    if not reads or not searches:return None
    read_ids={str(o.get("doc_id")) for o in reads}
    observed_text=" ".join(str(o.get("text","")) for o in reads).casefold()
    query_text=question.casefold()
    latest=searches[-1].get("hits",[])
    ranked=[]
    for rank,hit in enumerate(latest):
        doc_id=str(hit.get("doc_id",""))
        if not doc_id or doc_id in read_ids:continue
        title=str(hit.get("title","")).strip()
        title_norm=title.casefold()
        title_words=normalize(title).split()
        # Word-boundary title matching is only from observed text/question.
        relevance=(4.0 if len(title_words)>=2 and title_norm in observed_text else 0.0)
        relevance+=(2.0 if title_norm in query_text and title else 0.0)
        relevance+=sum(1 for word in title_words if len(word)>3 and word in observed_text)*0.3
        relevance+=sum(1 for word in title_words if len(word)>3 and word in query_text)*0.2
        ranked.append((-relevance,rank,doc_id))
    return min(ranked)[2] if ranked else None


class BridgeGroundedPolicy:
    def __init__(self,grounded_policy):
        self.base=grounded_policy
        self.name="bridge_"+grounded_policy.name
        self.model_name=getattr(grounded_policy,"model_name",None)
        self.adapter_path=getattr(grounded_policy,"adapter_path",None)
        self.last_role=""
        self.last_verification=None
        self.last_model_calls=0

    @property
    def model_calls(self)->int:return int(getattr(self.base,"model_calls",0))

    @property
    def prompt_tokens(self)->int:return int(getattr(self.base,"prompt_tokens",0))

    @property
    def completion_tokens(self)->int:return int(getattr(self.base,"completion_tokens",0))

    def _rule(self,action:Action,reason:str)->Action:
        self.last_role="bridge_planner"
        self.last_verification={"approved":False,"reason":reason}
        self.last_model_calls=0
        return action

    def decide(self,question:str,observations:list[dict],remaining:int)->Action:
        reads=[o for o in observations if o.get("kind")=="read"]
        searches=[o for o in observations if o.get("kind")=="search"]
        # Only once: a candidate second-hop search after the FIRST read.
        if len(reads)==1 and remaining>=3:
            last_read_idx=max(i for i,o in enumerate(observations)
                              if o.get("kind")=="read")
            has_search_since_read=any(o.get("kind")=="search"
                                      for o in observations[last_read_idx+1:])
            if not has_search_since_read:
                proposal=next_bridge_query(question,observations)
                if proposal:
                    return self._rule(Action("search",query=proposal.query),
                                      "gold_blind_second_hop_search")
            else:
                choice=choose_unread_hit(question,observations)
                if choice:
                    return self._rule(Action("read",doc_id=choice),
                                      "bridge_observed_frontier_read")
        action=self.base.decide(question,observations,remaining)
        self.last_role=getattr(self.base,"last_role","grounded")
        self.last_verification=getattr(self.base,"last_verification",None)
        self.last_model_calls=getattr(self.base,"last_model_calls",0)
        return action
