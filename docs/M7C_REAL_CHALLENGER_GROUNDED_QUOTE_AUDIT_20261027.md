# M7-C: real local Qwen3-8B Challenger, independent TRAIN quote audit (2026-10-27)

> **Research status: actual model inference completed; real independent source replay completed; NO new trustworthy task answers, NO multi-generation RSI training or improvement.** These are sequential TRAIN-only *candidate queues*, NOT learned model generations.

## Reproducibility / isolation

- Project: `YOUessi/Self-Improving_Deep_Research_Agent`; isolated branch `feat/m7-rsi-multi-generation`, draft PR #11. Source code snapshot for final experiment/CPU verifier `479843a46674269522cf6283b5522680d3139431`.
- Tang: **RTX 4090 Laptop 16GB**. Actual local **Qwen3-8B-AWQ**, weights in `/home/you/models/Qwen3-8B-AWQ`, served through local vLLM API on `127.0.0.1:18985`. `--max-model-len 3072`, `--gpu-memory-utilization 0.72`, non-thinking JSON mode. No external API, no DEV gold, no final-B160.
- Frozen HotpotQA **TRAIN-only 1,024-question subset**: `~/experiments/sidra_m4/data/train_oracle1024_seed43/converted`, `questions.jsonl` SHA256 `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`, documents SHA256 `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`. Original TRUSTED verifier maps hold source IDs and retrieval-difficulty proxy, but only **`{challenge_id,question}`** enters the plain Challenger model.
- Three frozen candidate-generation input batches, **24 distinct TRAIN questions each** with seeds **43,44,45**; source IDs, normalized text hashes and near-duplicate comparisons passed independent re-audit. Across 72 seeds: exactly **24 direct / 24 bridge / 24 hard** retrieval-proxy tasks. Difficulty denotes *TRAIN-only original BM25 support exposure*, not demonstrated reasoning difficulty.
- Actual initial direct 8B vLLM start failed: previous HF cache path held metadata but **not weights**. Repaired by using real on-disk `/home/you/models/Qwen3-8B-AWQ` weights. The installed vLLM version rejected `--disable-log-requests`; restarted without this unsupported flag. All problems logged in `~/experiments/sidra_rsi_m7/logs/`. After model inference, terminated only this worker's local vLLM service; GPU returned to ~349 MB idle.

## E-021: 72 **actual** model-generated unlabelled Challenger tasks

Model receives only TRAIN questions, no original gold answers or HotpotQA supporting document IDs. A JSON-only task generation prompt requests a different multi-hop question. Local Qwen3-8B-AWQ inference actually executed:

| Frozen input cohort | Generated responses | Syntax/novelty gate | Independently verified ANSWERS |
| --- | ---: | ---: | ---: |
| G1 TRAIN-24 / seed43 | 24 | 24/24 | **0** |
| G2 TRAIN-24 / seed44 | 24 | 24/24 | **0** |
| G3 TRAIN-24 / seed45 | 24 | 24/24 | **0** |
| **Total** | **72** | **72/72** | **0** |

The **72/72** result is only JSON / length / near-duplicate screening, NOT task truth or useful RL data. Some suggested questions carry false/unsupported premises or refer to an answer not reachable through the input's original documents.

Original candidate output: `~/experiments/sidra_rsi_m7/challenger/g{1,2,3}_train24_seed{43,44,45}_qwen3_8b_awq/` (TRAIN-only; not pushed to GitHub). Campaign integrity report: `~/experiments/sidra_rsi_m7/reports/actual_8b_awq_three_gen_72_candidate_audit.json`, generated via `scripts/audit_rsi_campaign.py`. It recomputes original TRAIN queues, trusted maps and cross-generation hashes, checks model-output structure a second time, and grants **zero** training labels.

## E-022: retrieval-grounded Qwen3-8B Challenger + independent literal quote Verifier

To address the evidence problem, created the second real-generation path:

1. Independently BM25 SEARCH the original source TRAIN question (same trusted corpus, bounded top-4, scope checks). The Challenger sees **only original TRAIN question + four actual retrieved snippets with tool document IDs**; never TRAIN gold, supporting-label IDs, DEV answers or any final-B items.
2. Model proposes a **new question, proposed answer, and two different original-document verbatim quotes**. It may abstain. The proposed answer must appear exactly in at least one quoted sentence. Model uses the same local Qwen3-8B-AWQ with non-thinking JSON output.
3. Immediately reject duplicate citation IDs, out-of-corpus citations, invented/edited quotes, malformed prompts, non-extractive proposed answers, and repeats of source question.
4. A **separate offline verifier process** reloads the saved TRAIN corpus, re-runs BM25 independently, recomputes packet SHA, and replays every accepted quoted passage against the source. Even when the quotes are exact, **DO NOT** promote answer to gold or train the Solver: semantic entailment has not been proved.

Actual local Qwen3-8B-AWQ generated **8 prompts from each** of the three same frozen TRAIN source queues (inference seeds 153,154,155; distinct from sampling seeds 43–45):

| Quote gate status | Cohort 1 (8) | Cohort 2 (8) | Cohort 3 (8) | Total |
| --- | ---: | ---: | ---: | ---: |
| Valid: two source quotes + extractive proposed answer | **4** | **3** | **1** | **8/24** |
| Answer not present literally in provided quotes | 3 | 1 | 0 | **4** |
| Invented/edited non-exact quote | 0 | 2 | 2 | **4** |
| Two citations not different/actually retrieved | 0 | 0 | 4 | **4** |
| Proposed question repeats source question | 0 | 2 | 1 | **3** |
| Model abstained | 1 | 0 | 0 | **1** |

The exact matching offline source-replay verifier independently confirmed **4+3+1=8** structurally quote-backed outputs, with **0/24 independently semantically validated answers** and **0 new training labels**. These are *different metrics*: the 72/72 plain JSON validity rate **cannot** be compared as factual correctness with the 8/24 strict quote structural rate.

Example failure uncovered during manual evidence inspection: the model asked for the population of the city where Batdorf & Bronson coffee roaster is based. One quote identifies the company as **Olympia, Washington–based**; the second quote gives the population of **Bronson, Michigan**. A matching name fragment let it supply **2,349**, an unsupported answer for Olympia. Other proposals cite a soundtrack associated with the film `Batman` but independently cite Michael Keaton starring in a **different film**; both quotations are real but the proposed two-hop inference is not established. Thus even **8 quote-valid candidates should be treated as *UNLABELLED*.**

### Artifacts and code

- Model samples: Tang `~/experiments/sidra_rsi_m7/challenger/g{1,2,3}_grounded8_qwen3_8b_awq_v1`. Independent quote reports: Tang `~/experiments/sidra_rsi_m7/reports/g{1,2,3}_grounded8_8b_independent_quote_audit.json`.
- `src/sidra/rsi_grounded_challenger.py`: explicit gold-redacted snippets, two-doc quote gate, local-only backend.
- `scripts/run_rsi_grounded_challenger.py`: actually invoked local model, saved outputs and model-count summaries (not weights).
- `src/sidra/rsi_grounded_audit.py`, `scripts/audit_rsi_grounded_candidates.py`: **independent** offline tool observation / source text replay, accepts no model-declared training labels.
- `src/sidra/rsi_campaign_audit.py`, `scripts/audit_rsi_campaign.py`: cross-generation frozen-queue uniqueness and leakage checks, candidate totals and hashes.
- `tests/test_rsi_grounded_challenger.py`, `tests/test_rsi_campaign_audit.py`: strict falsified quote, duplicate document, forged summary, file tampering, prompt-label leakage, cross-generation repetition, no training approval.
- Final Tang pinned snapshot `479843a46674269522cf6283b5522680d3139431`: **147/147 CPU tests passed**. GitHub Actions on same head **success**. No raw benchmark files, checkpoint weights, secrets or original private files uploaded.

## Key algorithmic finding and necessary next work

**Research gap is factual premise and multi-hop relation verification, not JSON output rate.** A valid answer string in one real quote + another real citation cannot show the proposed question logically follows. In particular, adjacent but distinct named entities (e.g. a **brand** containing a city name versus an actual **city** with that name) defeat naive quote matching.

Before these candidate questions can enter a Trainer, implement and independently evaluate:
- **Semantic premise graph:** extract claimed entities/relations and source-specific roles for each hop; require correct bridge entity disambiguation and evidence-based answer uniqueness rather than mere lexical overlap. A separate Verifier must be able to return `unsupported` with explicit reasons.
- **Independent answer verification:** TRAIN-only factual references and original HotpotQA sentence-level facts, or independently annotated novel task labels. Do not inherit source question gold; do not accept Challenger's self-reported proposed answer as ground truth.
- **A genuine model update loop:** next-generation Solver must produce fresh `Search/Read/Finish` traces on verified TRAIN tasks, reward must be independently replayed, Trainer must actually update weights using on-policy action likelihood, and next generation must use that new checkpoint. Paired open-pilot DEV + predeclared independent eval-A, >=3 actually independent optimizer seeds, cost and confidence intervals are necessary.
- **Checkpoint gate:** previous one-step online GRPO LoRA SHA change exists, but QA pilot32 showed **zero** verified and EM improvement; existing RSI ledger rejected it and retained parent checkpoint. No new model promoted here. **final-B160 remains sealed**.

The final outcome of this round is **real model-driven challenge generation plus independently verified source-quote integrity, exposing serious semantic-bridge errors** — not a claimed RSI improvement.
