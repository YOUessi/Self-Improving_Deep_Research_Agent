"""Independent model-action I/O to tool-trajectory consistency checks.

E-030 optionally permits an explicitly declared deterministic, gold-blind
single READ action, independently replayed from prior observed tool state.
Every OTHER action must still match an actual model reply, context and seed.
This is not cryptographic attestation of model serving origin or likelihood.
"""
from __future__ import annotations

import json

from sidra.agent import AgentConfig
from sidra.policies import _json_object
from sidra.rsi_observed_read import choose_observed_second_read
from sidra.schema import Action, AgentTrace


def audit_model_action_io(
    traces: list[AgentTrace], model_io: list[dict], *,
    seed: int, config: AgentConfig,
    allow_observed_read_guard: bool = False,
) -> dict:
    if not traces or len(traces) != len(model_io):
        raise ValueError("missing/unpaired model action I/O transcript")
    if type(allow_observed_read_guard) is not bool:
        raise ValueError("invalid opt-in rule guard flag")
    total = 0
    prompt_tokens = 0
    completion_tokens = 0
    legal_actions = 0
    terminal_action_errors = 0
    rule_reads = 0
    for tr, episode in zip(traces, model_io, strict=True):
        if set(episode) != {"challenge_id", "calls"}:
            raise ValueError("model I/O episode contains unexpected fields")
        if episode["challenge_id"] != tr.question_id:
            raise ValueError("model I/O identity mismatch")
        if (tr.policy.startswith("optin_read_before_finish_")
            != allow_observed_read_guard):
            raise ValueError("guarded policy identity/manifest mismatch")
        calls = episode["calls"]
        rule_steps = [s for s in tr.steps
                      if isinstance(s, dict)
                      and s.get("role") == "observed_read_before_finish"]
        if len(rule_steps) > 1:
            raise ValueError("multiple deterministic READ guard actions")
        if (not isinstance(calls, list) or not calls
            or len(calls) != tr.model_calls
            or len(calls) != len(tr.steps) - len(rule_steps)):
            raise ValueError("model I/O / SearchAgent call-count mismatch")
        observations = []
        ep_prompt = 0
        ep_completion = 0
        model_index = 0
        for step_no, step in enumerate(tr.steps):
            if not isinstance(step, dict):
                raise ValueError("malformed SearchAgent tool step")
            if step.get("role") == "observed_read_before_finish":
                if not allow_observed_read_guard:
                    raise ValueError("unapproved deterministic tool step")
                if (step.get("role_model_calls") != 0
                    or step.get("verification") != {
                        "approved": True,
                        "reason": "observed_only_single_extra_read",
                        "legal_frontier_size": (
                            step.get("verification", {}).get("legal_frontier_size")
                            if isinstance(step.get("verification"), dict)
                            else None
                        ),
                        "gold_used": False,
                    }):
                    raise ValueError("READ guard attestation format mismatched")
                try:
                    proposal = choose_observed_second_read(
                        tr.question, observations,
                        max_steps=config.max_steps,
                        max_tool_calls=config.max_tool_calls
                    )
                except (ValueError, TypeError) as exc:
                    raise ValueError("READ guard tool observation forgery") from exc
                if proposal is None:
                    raise ValueError("READ guard not eligible in observed state")
                if (step["verification"]["legal_frontier_size"]
                    != proposal.visible_candidates):
                    raise ValueError("READ guard frontier size forged")
                try:
                    action = Action.from_dict(step.get("action", {}))
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError("invalid deterministic READ action") from exc
                canonical = json.loads(json.dumps(action.to_dict()))
                if (action.kind != "read" or action.doc_id != proposal.doc_id
                    or action.query or action.answer or action.citations
                    or canonical != step.get("action")):
                    raise ValueError("READ guard did not choose observed legal document")
                if not isinstance(step.get("observation"), dict):
                    raise ValueError("missing rule READ tool observation")
                observations.append(step["observation"])
                rule_reads += 1
                continue
            if allow_observed_read_guard:
                # Model steps may only inherit the benign role set by the
                # wrapper. A forged rule tag or zero-call annotation fails.
                if step.get("role") not in (None, "solver"):
                    raise ValueError("unexpected model-tool role")
                if step.get("role") == "solver" and step.get("role_model_calls") != 1:
                    raise ValueError("model decision claimed wrong number of calls")
            if model_index >= len(calls):
                raise ValueError("missing model response for non-rule action")
            call = calls[model_index]
            model_index += 1
            if not isinstance(call, dict):
                raise ValueError("malformed model response record")
            messages = call.get("messages")
            if (not isinstance(messages, list) or len(messages) != 2
                or not isinstance(messages[0], dict)
                or not isinstance(messages[1], dict)
                or messages[0].get("role") != "system"
                or messages[1].get("role") != "user"
                or not isinstance(messages[0].get("content"), str)):
                raise ValueError("missing original model prompt messages")
            try:
                user = json.loads(messages[1]["content"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("non-JSON or missing model user context") from exc
            expected = {
                "question": tr.question,
                "remaining_steps": config.max_steps - step_no,
                "tool_history": observations,
            }
            if (user != expected or not isinstance(user, dict)
                or set(user) != set(expected)):
                raise ValueError("model prompt deviates from goldblind replayed tools")
            if call.get("constrained_schema") is not True:
                raise ValueError("solver decoder restriction flags inconsistent")
            if (type(call.get("generation_seed")) is not int
                or call["generation_seed"] != seed + total):
                raise ValueError("inference seed sequence differs from run manifest")
            for name in ("prompt_tokens", "completion_tokens"):
                if type(call.get(name)) is not int or call[name] < 0:
                    raise ValueError("invalid API usage telemetry")
            raw = call.get("raw_completion")
            if not isinstance(raw, str):
                raise ValueError("model action text missing")
            try:
                generated = Action.from_dict(_json_object(raw))
            except (ValueError, TypeError, KeyError):
                generated = None
            if "action" in step:
                if generated is None:
                    raise ValueError("model action does not parse although trace claims action")
                normalized = json.loads(json.dumps(generated.to_dict()))
                if normalized != step["action"]:
                    raise ValueError("model output/action in SearchAgent trace mismatch")
                legal_actions += 1
            else:
                if (step_no != len(tr.steps) - 1 or tr.termination != "error"
                    or "error" not in step):
                    raise ValueError("unattributed model action output in tool trace")
                terminal_action_errors += 1
            if "observation" in step:
                if step.get("action", {}).get("kind") not in ("search", "read"):
                    raise ValueError("observation was not produced by a tool")
                observations.append(step["observation"])
            ep_prompt += call["prompt_tokens"]
            ep_completion += call["completion_tokens"]
            total += 1
        if model_index != len(calls):
            raise ValueError("unused recorded model API response")
        if (ep_prompt != tr.prompt_tokens or ep_completion != tr.completion_tokens):
            raise ValueError("trace model token counters disagree with server responses")
        prompt_tokens += ep_prompt
        completion_tokens += ep_completion
    return {
        "verified_episode_IO_count": len(traces),
        "verified_response_count": total,
        "matched_successful_model_actions": legal_actions,
        "verified_deterministic_observed_READs": rule_reads,
        "terminal_model_or_tool_error_responses": terminal_action_errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "all_model_contexts_and_seeds_match_tool_replay": True,
        "model_origin_cryptographically_attested": False,
        "limits": "Mutable local records cannot prove network origin/model weights or "
                  "guided-decoder token probabilities. No on-policy optimizer proof.",
    }
