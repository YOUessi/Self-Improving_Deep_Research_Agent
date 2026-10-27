"""E-027: independent model-action I/O to tool-trajectory consistency checks.

This checks a saved inference transcript against the model-facing prompts,
the actual tool observation chain and assistant actions. It catches forged
prompt contexts, inserted gold label fields, skipped or reordered model
outputs, and seed/tokens mismatches. It is NOT cryptographic proof that the
named model checkpoint really produced an API response.
"""
from __future__ import annotations

import json
from sidra.agent import AgentConfig
from sidra.policies import _json_object
from sidra.schema import Action,AgentTrace


def audit_model_action_io(
    traces:list[AgentTrace],model_io:list[dict],*,
    seed:int,config:AgentConfig)->dict:
    if not traces or len(traces)!=len(model_io):
        raise ValueError("missing/unpaired model action I/O transcript")
    total=0
    prompt_tokens=0
    completion_tokens=0
    legal_actions=0
    terminal_action_errors=0
    for tr,episode in zip(traces,model_io,strict=True):
        if set(episode)!={"challenge_id","calls"}:
            raise ValueError("model I/O episode contains unexpected fields")
        if episode["challenge_id"]!=tr.question_id:
            raise ValueError("model I/O identity mismatch")
        calls=episode["calls"]
        if (not isinstance(calls,list) or not calls
            or len(calls)!=tr.model_calls or len(calls)!=len(tr.steps)):
            raise ValueError("model I/O / SearchAgent call-count mismatch")
        observations=[]
        ep_prompt=0
        ep_completion=0
        for step_no,(step,call) in enumerate(zip(tr.steps,calls,strict=True)):
            if not isinstance(call,dict):
                raise ValueError("malformed model response record")
            messages=call.get("messages")
            if (not isinstance(messages,list) or len(messages)!=2
                or not isinstance(messages[0],dict)
                or not isinstance(messages[1],dict)
                or messages[0].get("role")!="system"
                or messages[1].get("role")!="user"
                or not isinstance(messages[0].get("content"),str)):
                raise ValueError("missing original model prompt messages")
            try:
                user=json.loads(messages[1]["content"])
            except (TypeError,ValueError,KeyError) as exc:
                raise ValueError("non-JSON or missing model user context") from exc
            expected={
                "question":tr.question,
                "remaining_steps":config.max_steps-step_no,
                "tool_history":observations,
            }
            if user!=expected or not isinstance(user,dict) or set(user)!=set(expected):
                raise ValueError("model prompt deviates from goldblind replayed tools")
            if call.get("constrained_schema") is not True:
                raise ValueError("solver decoder restriction flags inconsistent")
            if type(call.get("generation_seed")) is not int or call["generation_seed"]!=seed+total:
                raise ValueError("inference seed sequence differs from run manifest")
            for name in ("prompt_tokens","completion_tokens"):
                if type(call.get(name)) is not int or call[name]<0:
                    raise ValueError("invalid API usage telemetry")
            raw=call.get("raw_completion")
            if not isinstance(raw,str):
                raise ValueError("model action text missing")
            try:
                generated=Action.from_dict(_json_object(raw))
            except (ValueError,TypeError,KeyError):
                generated=None
            if "action" in step:
                if generated is None:
                    raise ValueError("model action does not parse although trace claims action")
                # dataclass asdict() retains tuples; JSONL tuples become lists.
                normalized=json.loads(json.dumps(generated.to_dict()))
                if normalized!=step["action"]:
                    raise ValueError("model output/action in SearchAgent trace mismatch")
                legal_actions+=1
            else:
                if (step_no!=len(tr.steps)-1 or tr.termination!="error"
                    or "error" not in step):
                    raise ValueError("unattributed model action output in tool trace")
                # SearchAgent intentionally withholds action on terminal error.
                terminal_action_errors+=1
            if "observation" in step:
                if step.get("action",{}).get("kind") not in ("search","read"):
                    raise ValueError("observation was not produced by a tool")
                observations.append(step["observation"])
            ep_prompt+=call["prompt_tokens"]
            ep_completion+=call["completion_tokens"]
            total+=1
        if (ep_prompt!=tr.prompt_tokens or ep_completion!=tr.completion_tokens):
            raise ValueError("trace model token counters disagree with server responses")
        prompt_tokens+=ep_prompt
        completion_tokens+=ep_completion
    return {
        "verified_episode_IO_count":len(traces),
        "verified_response_count":total,
        "matched_successful_model_actions":legal_actions,
        "terminal_model_or_tool_error_responses":terminal_action_errors,
        "prompt_tokens":prompt_tokens,
        "completion_tokens":completion_tokens,
        "all_model_contexts_and_seeds_match_tool_replay":True,
        "model_origin_cryptographically_attested":False,
        "limits":"Mutable local records cannot prove network origin/model weights or "
                 "guided-decoder token probabilities. No on-policy optimizer proof.",
    }
