# Reuse and multi-agent design (2026-10-26)

## Existing projects

- **CodeRL-Lab**: [reward](https://github.com/YOUessi/CodeRL-Lab/blob/main/src/coderl_lab/reward.py), [GRPO](https://github.com/YOUessi/CodeRL-Lab/blob/main/src/coderl_lab/train/grpo.py): reuse train-only split isolation, rollout logging, controlled comparison, reward-variance analysis and experimental methodology. The Python code executor's public-tests reward **must not** be used to score Search Agent evidence. Coding LoRA adapters do not constitute search-specific training.
- **OpenCollab**: [workflow](https://github.com/YOUessi/OpenCollab/blob/main/opencollab/application/workflow.py): reuse planner/researcher/solver roles, budget accounting and trace attribution design. No full OpenCollab dependency or unconstrained parallel calls in M2.
- **OpenCollab Execution Server**: sandbox/checkpoint concepts are reserved for future code/browser tool use, not needed for local BM25.
- **Agentic GPU Debugger**: only future GPU efficiency/benchmark reuse; not part of the search loop.

## M2 role design

`MultiAgentPolicy` uses Planner (first query), Researcher (search and read), Solver (answer and citations), and independent EvidenceAudit (read-before-cite, minimum sources, lexical presence). One model instance can serve multiple distinct role prompts, avoiding multiple copies on Tang's 16GB GPU. Roles are sequential, **not concurrent independent LLM processes yet**.

All tools go through the existing SearchAgent limits. No evaluator answers, supporting-document IDs or hidden test labels enter any role prompt. EvidenceAudit is a *weak* source-only gate, not semantic entailment or benchmark correctness. Record role traces, counts and tool budgets; compare against a single-agent baseline with the same tools plus inference cost.

## Qwen model audit on Tang

Qwen3-8B and Qwen3-8B-AWQ: snapshot directories contain only `config.json`; **no model weights**. Not usable. Qwen3-0.6B-Base and Qwen3-1.7B-Base: complete local weights, but these are base models rather than instruction-tuned search agents. Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct: complete, older generation, valid reference controls. vLLM, AutoAWQ, PEFT and TRL were not installed in the checked Tang Python. Hugging Face HTTPS probes timed out.

Run `python scripts/audit_models.py` to distinguish model directories from complete weights. Do not silently download, modify system-wide packages, or label Qwen3 Base inference a fair comparison with Qwen2.5-Instruct.

## Gates

M2 CPU/role regression -> real multi-hop benchmark -> full instruction-tuned Qwen3 model selection -> frozen single-vs-multi baseline -> verifier-filtered SFT -> GRPO -> recursive Challenger/Solver/Verifier iterations. Current role specialization is NOT RSI training.
