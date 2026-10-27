# M7 V5 / Hybrid: executed outcomes on original HotpotQA TRAIN cohort — 2026-10-27

## Executive outcome

**V5 finite answer catalog was integrated correctly but failed end-to-end Agent quality gates.** A separate 8B answer + TRAIN-learned evidence attribution hybrid achieved a positive module-level result on 64 *question-disjoint* original HotpotQA TRAIN examples. This **does not** establish that V5 online GRPO helps, full-Wikipedia/realtime-web search generalizes, or RSI is complete.

### A. V4/V5 action-space and exact probability contract

- Existing V4 legal action policy kept as an untouched causal baseline. V5 loads `finish_candidates_v5.visible_finish_options`, interleaves two actually READ texts, supplies polarity-neutral yes/no options, keeps at most **10** possible choices `0..9` (each exactly ONE tokenizer token), and reserves legal Search/READ plus mandatory abstention, respecting max 6 steps and 5 tool calls. Action sampling and log-prob in `FiniteActionPolicy` use the same legal-choice mask + observation-only action biases.
- Code passed **268 CPU tests**. V4/V5 finite correct-answer coverage under posthoc *perfect READ of gold supporting passages* from original TRAIN1024 was **403/1024 → 564/1024**, including binary questions **0/51 → 48/51**. Previously seen DEV pilot32: **9/32 → 12/32**. This is **only an oracle source-exposure ability ceiling**, not an Agent benchmark.
- Preregistered **real Qwen3-0.6B stop-SFT Agent**, same checkpoint and BM25 tools, DEV pilot32 repeated paired seed151/157: V4 answer EM **6/64 episodes (9.375%)**, V5 **4/64 (6.25%)**; strict verification V4 **4/64**, V5 **2/64**. Because the 32 original questions repeat at two stochastic seeds, resampling clusters by **32 QUESTION IDs**, not 64 independent questions. Paired EM delta V5−V4 **−3.125 percentage points**, question-cluster 95% bootstrap [−10.9375,+6.25]pp; strict −3.125pp, CI [−7.8125,0]pp. Mean prompt tokens **4,837.1→5,215.1** per task (+378), latency ~254→277ms; V5 did not gain quality.
- **Decision: Do not launch V5 GRPO.** The legal action-space extension alone increases representability but not learned correct action selection.

### B. TRAIN-only evidence ranker (not a language-model weight update)

- Original TRAIN1024 question/answer/support labels: 838 question groups/8,377 document rows fitted; 186 separate TRAIN question groups/1,852 documents used to choose threshold. Weighted NumPy logistic regression, 12 lexical/document features, 400 steps, lr0.07, threshold selected **0.2** on TRAIN internal validation, not public DEV. Saved SHA256 `2d7e1089d83f377a91700645165ae5120bd830c7f76674a38adc02c731b3f717`; serving consumes ONLY question, proposed answer and **actually READ** document texts, no gold IDs or unseen corpus. Ranker always picks two distinct observed sources or abstains; quotes copied as text snippets.
- Posthoc *previously viewed* DEV pilot32, using frozen, previously generated Qwen3-8B-AWQ answers: A8 source traces **13/32 raw answer EM → 11/32 kept answer EM, 8/32 strictly supported answer+doc citations**. Original SFT source traces **12/32 → 10/32, strict 8/32**. It uses **zero extra LLM calls** over already stored 8B outputs. These are not prospective heldout results.

### C. Prospective new-question original TRAIN64 test — genuinely executed

**Data freeze and integrity (before model calls):**

- Official original TRAIN90,447 source Parquet files checksum verified. SHA256(seed 71212026: original question ID) ranking, exclude **all 1,024 prior TRAIN questions** and 256 public DEV IDs. Freeze exactly 64 evaluation-only TRAIN QA plus original distractor documents. New ordered question-ID digest `13a5b42906d29640d2be1cd81adc0bf72cb48b49a38e7f51554c99c4cda10565`. Questions SHA `3ab9ff0204991641e306323c3adab792c5b5d68d2f4df58f64490536523d01c4`; docs SHA `425be36c024ce06ba64a4a36ba1caa52dba147c5a9d253f9548b903ef829864d`. This is an **independent question sample within the original TRAIN distribution**, NOT an independent external unseen-domain dataset.
- Further read-only cross-workstream contamination scan examined 554 existing text/json/csv output files over SIDRA RSI and evidence experimental result directories; no saved occurrence of any of the fresh 64 question IDs in those files. This is a bounded artifact scan, not a proof against unseen unsaved exposures.
- Source code for actual second-stage evaluation: **`f3e0432cf2a4d0486a7ab9a725319abb6509f10a`**, **280 CPU tests passed** before execution. Existing V4 frozen SearchAgent stage-1 dataset and ranker SHA checked at runtime; no re-training on the new64.

**Phase 1: real Qwen3-0.6B-Base stop-SFT SearchAgent evaluation** on new64: **0/64 answer EM**, **0/64 strict answer+citation verification**, mean original support cited recall **0.601562**, mean **4.625** tool calls (1.875 search, 2.75 READ), 5.625 model calls, ~246.94ms/task.

**Phase 2: frozen Qwen3-8B-AWQ answers + TRAIN-only ranker**, on *exactly Phase 1's stored real Search/Read observations*, run on Tang with loopback vLLM GPU0 shared lock and process-owned shutdown trap. Of 64 tasks, 62 had >=2 distinct READ sources and received 8B calls, 1 solver/parse error, 30 abstentions after ranker (including insufficient original evidence):

| Metric | Original finite V4 real SearchAgent | 0.6B real Search/Read → 8B answer → TRAIN evidence ranker |
| --- | ---: | ---: |
| Answer EM | **0/64** | **19/64** |
| Answer EM before ranker abstention | — | **27/64** |
| Strict answer & original passage citation | **0/64** | **9/64** |
| Answer & original supporting-sentence presence in cited actual READ | **0/64** | **9/64** |
| Mean cited gold-support recall | **0.601562** | **0.359375** |
| Mean actual Search/Read tool calls | 4.625 | 4.625 (unchanged) |
| Mean total model calls | 5.625 | **6.59375** |
| Mean wall time | 246.94ms | **908.47ms** |

- Extra GPU calls: **62** Qwen3-8B requests, 35,363 input tokens + 2,672 output tokens, **42.338s** total 8B inference wall time.
- Question-cluster **5,000 bootstrap** 95% CI of positive strict success difference: **[+6.25pp,+23.4375pp]**, answer EM improvement **[+18.75pp,+40.625pp]**. This is descriptive for this sample and low baseline; not guaranteed out-of-domain gains.
- Original gold support sentences checked OFFLINE after all 64 independent 8B outputs were generated; gold answers/passages did not enter model/selector requests.
- The ranker is **fail closed**: may reduce answer EM from 27 to 19 via abstentions and support recall 60.2%→35.9%. Thus it does NOT solve relevance/coverage even though strict success rose vs weak original baseline.
- No new LM weight updates, no on-policy GRPO optimizer during this hybrid evaluation, no proven semantic entailment or adversarial prompt-injection safety. **Two-phase modular system** over frozen actual tool trajectories, **not one streaming end-to-end Agent**.
- Tang result path: `~/experiments/sidra_m7_v5/results/M7_HYBRID_FRESH_TRAIN64_s151_f3e0432c/`. Model and ranker artifacts stay on Tang; GitHub commits scripts/tests and this non-sensitive aggregate report only.
- At completion `2026-10-27 14:30:46+08:00`, vLLM received SIGTERM from its owning experiment; parent+engine exited, port18988 closed, GPU VRAM returned to ~350MiB used / 15,570MiB free.

**Decision / research roadmap gate:** V5 is rejected as a stand-alone legal-action replacement on current model, so **do not run V5 on-policy GRPO** merely because coverage rose. The 8B answer + citation-ranker module *passes a positive question-disjoint TRAIN-only engineering feasibility test*; next implement a true streaming integration, jointly track 8B token/cost and semantic sentence entailment, and preregister another truly unused dataset BEFORE final model selection. Only if a TRAIN-only on-policy optimizer improves a fixed independent quality gate should it proceed to RSI checkpoint generation. Sealed DEV final-B160 stays unopened.

### Implementation faults encountered and corrected

1. TRAIN ranker input has two source documents for 1/1024 cases; the first trainer assumed at least three sources. Modified its TRAIN-only data loader to accept >=2, and skip distractor-required internal evaluation for samples without a distractor. Tests passed; no source files altered.
2. New frozen64 cohort test contained a mismatched closing bracket (syntax error), corrected and re-ran full tests. Another initial sandbox snapshot command omitted setting `sha`, corrected by using already existing pinned snapshot after read-only SHA verification.
3. Previous quote/sentence-ID 8B pilot had mistaken JSON response schema reuse, now fixed in prior work; do not attribute parse-contract failures to model reasoning failures.
4. GPU contention from previous RSI E027 idle vLLM was solved by shared advisory lock, free-VRAM checks and `trap` shutdown. New64 8B server terminated properly without affecting foreign jobs.

**Bottom line:** This is genuine engineering progress on a deployable modular research stack, but **not** completed RSI / self-improving GRPO algorithm research; strictly separate SFT/GRPO model gains from replacing a 0.6B finite answerer with an 8B conditioned solver.
