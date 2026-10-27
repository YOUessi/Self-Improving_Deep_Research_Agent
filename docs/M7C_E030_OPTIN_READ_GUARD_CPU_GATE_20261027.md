# M7-C E-030 — Opt-in observed-only second-READ guard: CPU engineering PASS, model QA NOT RUN

**2026-10-27 Tang +08:00.** Branch `feat/m7-rsi-multi-generation`, draft PR #11. The test and model-only observational feasibility code SHA is **`299790c4c236aeb5e076b5b08b998d72f150e5c8`**. E-030 was preregistered separately in `M7C_E030_READ_BEFORE_FINISH_PREREG_20261027.md`; no real model QA was performed in this stage.

## Why E-030 was needed

E-029 independently replayed the 72 genuine frozen original HotpotQA TRAIN Qwen3-0.6B STOP-LoRA Solver episodes and verified a specific structural failure mode:

- **71/72** real model trajectories executed exactly one READ;
- **1/72** executed zero READ;
- **0/72** executed a second READ, preventing a fair retrospective same-action-slot replacement test;
- Historical original answer EM = **17/72**, answer+complete original support citations = **0/72**. They remain old E-027 outcomes, **not E-030 outcomes**.

Thus a second-READ intervention would add a **real READ action**, not just choose a better document in an existing READ slot. The research protocol explicitly counts that tool cost under the same global budget.

## New implementation

1. `src/sidra/rsi_read_before_finish.py` adds an **opt-in** `ObservedReadBeforeFinishPolicy` wrapper. It acts only after exactly one genuine READ, uses already-seen SearchAgent hit IDs/title/snippet and prior READ text, and inserts at most one READ if 2+ unread observed IDs and enough original step/tool budget remain. It never reads original TRAIN labels/Verifier maps or unseen full corpus. The model subsequently continues on the **actual** new tool observation.
2. `run_real_solver(...,use_observed_read_guard=True)` and `scripts/run_rsi_original_solver.py solve --observed-read-before-finish` expose the option. Default remains **off**; base real-model I/O is separately saved. Both arms keep `max_steps=6`, `max_tool_calls=5`, `search_top_k=3`, with no free extra tool budget.
3. `src/sidra/rsi_model_io_audit.py` independently recomputes the deterministic READ from model-visible observation history, verifies legal opaque doc ID, role/tool-call zero model tokens, exact rule metadata, and **still authenticates each normal model-generated action against the full original inference response, context, token count and seed**. A forged or undeclared rule action fails closed.
4. `src/sidra/rsi_solver_cohort.py` requires the corresponding guarded-run manifest declaration, and `rsi_solver_campaign_scorecard.py` rejects mixing guarded and unguarded cohorts under a single comparable arm.
5. `scripts/probe_rsi_guard_trigger_model_only.py` is a separate aggregate-only, **model-visible data only** retrospective feasibility probe. It cannot receive TRAIN gold, source question ID mappings or any external evaluation data.

## Verified result — CPU, no GPU model claims

| Engineering evidence | Outcome |
| --- | ---: |
| Full source CPU pytest at pinned SHA | **240 / 240 PASS** |
| Python compileall | **PASS** |
| GitHub Actions CPU Python 3.10 and 3.12 | **PASS / PASS** |
| Synthetic SearchAgent before/after wrapper | one additional legal READ, same global max tool/step caps |
| Malformed role/forged selected doc, context, model action/seed | rejected in automated tests |
| Older unguarded E-027 real-trace replay with new auditor | **72/72 valid; 17/72 answer EM; 0/72 strict** |
| Model-only feasibility, original 72 real traces | **71/72** would offer one observed legal second READ |
| **New live Qwen3 rollouts after the added READ** | **0** |
| Optimizer updates / authenticated RSI generations | **0 / 0** |

The 71/72 counterfactual **trigger** count is not an improved answer, successful Verifier result or live model continuation. It is calculated from *only the already-observed first-READ tool states*, with no TRAIN labels, using the frozen selector; it cannot establish whether the base model would successfully use the extra evidence.

### Tang immutable audit references

- Tested source: `~/experiments/sidra_rsi_m7/source_299790c4c236aeb5e076b5b08b998d72f150e5c8/`.
- 240/240 pytest: `~/experiments/sidra_rsi_m7/logs/E030_cpu_full_299790c4.log`.
- Model-only 3-cohort trigger result: `~/experiments/sidra_rsi_m7/reports/E030_MODEL_ONLY_GUARD_TRIGGER_299790c4.json`; SHA256 `e63b2c5602793d3207d8fc5aa2335b6521d70219dc5cd4c51f5c498ca488ac0e`.
- Re-audited old E-028 72-trace result with updated rule-aware verifier: `~/experiments/sidra_rsi_m7/reports/E030_E028_BACKCOMPAT_SOURCE_df5c9713.json`; SHA256 `6ea58415f8434dd55f157552606bada636da4b046c9b1808164f7eada56253f7`.
- Re-audit log and model-only probe logs under `~/experiments/sidra_rsi_m7/logs/`. No private question/doc/gold or model weight artifacts on GitHub.

## GPU nonpreemption and next critical experiment

Before and after the CPU work Tang RTX 4090 Laptop 16 GiB had **9,821 MiB occupied, 6,099 MiB free** by an independent existing `vllm serve` process (parent PID 86713, engine 86847). We did not interrupt it or launch a competing GPU job. Code was developed/pushed to GitHub and tested CPU-only on pinned Tang copies.

**Next meaningful validation:** first freeze a prospective disjoint original TRAIN evaluation cohort after excluding *all* prior M5 SFT data-exposure, M6/M7 training, E-026 real Solver questions and reserved holdouts; evaluate actual independent baseline-vs-guarded Qwen3 Search/Read/Finish trajectories with two matched stochastic seeds when the GPU is uncontended. Score strict answer+original support, answer EM, source READ recall, latency, calls, budget and paired uncertainty; keep the guard disabled if gains fail. No DEV/eval-A/final-B peeking. This is prerequisite evidence-retrieval improvement work, **NOT yet Agentic GRPO or multi-generation RSI**.

Only genuine matched-behavior online RL updates, per-generation checkpoint SHA/rollback and independent multi-seed gains can justify later RSI claims. E-030 has **none** of those outcomes.
