# M7 Hybrid answer+evidence independent question cohort protocol (frozen before inference)

## Why this branch changed from V5 to an answer module

V4→V5 expanded finite answer visibility on original TRAIN1024 (403→564 / 1024) but true same-checkpoint DEV pilot32 model runs at seeds151 and157 showed **no** answer/strict evidence gain. Paired question-cluster EM delta was −0.03125 with 95% CI [−0.109375,0.0625]; strict success delta −0.03125 with CI [−0.078125,0]. Prompt tokens +378/question. **No V5 GRPO has been authorized or run on these results.**

The investigation located two distinct barriers: the small finite learner cannot choose answers omitted by its catalog, and a stronger Qwen3-8B answer model can often answer correctly using actually READ text but fails to cite all original support passages.

### Frozen TRAIN-only evidence attribution learner

- Only original 1,024 HotpotQA TRAIN scoped questions, SHA256 of input questions `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`. 838 question groups for weighted logistic training (8,377 passage examples), 186 other question groups for internal threshold selection (1,852 passage examples). Labels are original TRAIN supporting-doc IDs.
- 12 deterministic features based on question/answer/title and actual READ text; no runtime gold fields or unread search snippets. NumPy training 400 steps, lr0.07, internal cutoff 0.2. Trained model SHA256 `2d7e1089d83f377a91700645165ae5120bd830c7f76674a38adc02c731b3f717`. Inference uses pure Python, no external labels.
- **Development-only posthoc pilot:** Original 8B generated 13/32 answer EM (A8 stored traces) and 12/32 (original SFT traces), with 0/32 complete citations in either. TRAIN-learned ranker chooses two actually READ sources (not forcing any unrelated citation), now scored 11/32 EM and 8/32 strict passage success on A8, 10/32 EM and 8/32 strict on original SFT. This is an offline re-score on **previously viewed** DEV pilot32 with **no new model inference**; it is not an independent end-to-end result.

## Newly frozen evaluation cohort before further model tuning

To avoid reusing all M7 training/teacher/evidence examples from the original TRAIN1024, a **new 64-question evaluation-only cohort** was frozen directly from the official 90,447 original TRAIN Parquet rows. Selection is SHA256(seed=71212026, original question ID), excluding **all 1,024** prior SFT/GRPO/ranker TRAIN question IDs plus all **256** frozen public DEV IDs, with official Parquet checksums verified. No answer, supporting title or model outcome was used to select question IDs.

- Frozen cohort path on Tang: `~/experiments/sidra_m7_v5/data/frozen_unseen_train64_s71212026/`.
- Ordered question ID SHA256 `13a5b42906d29640d2be1cd81adc0bf72cb48b49a38e7f51554c99c4cda10565`.
- Converted question JSONL SHA256 `3ab9ff0204991641e306323c3adab792c5b5d68d2f4df58f64490536523d01c4`.
- Converted document JSONL SHA256 `425be36c024ce06ba64a4a36ba1caa52dba147c5a9d253f9548b903ef829864d`.
- This is a **new question cohort from original TRAIN**, **NOT an external held-out dataset or official TEST**. It must remain excluded from all future model/ranker training, reward shaping and hyperparameter selection.

## Prospective next evaluation — no additional answer tuning

1. Run the *unchanged* Qwen3-0.6B-Base + original stop SFT LoRA through **V4** real BM25 SearchAgent on the fresh 64 questions, seed151, 6 steps, 5 tools, max context2048. Save immutable read/search traces with model and data SHA. **Do not use V5** (failed pilot go/no-go gate).
2. Stop the 0.6B process, then start the known verified Qwen3-8B-AWQ vLLM with a shared GPU lock, a loopback-only endpoint, and a process-owned shutdown trap; do not compete for memory.
3. Feed only genuinely READ text and question to the previously fixed, non-thinking Qwen3-8B-AWQ `answer+citations` JSON solver. Ignore its freeform citation claims; pass its answer to the frozen **TRAIN-only** evidence ranker, which may select two actually read sources and quote literal evidence, or abstain. Do not train either model on the new 64 questions.
4. Compare original V4 finite answer, raw 8B answer (without enforceable citations), and fail-closed hybrid answer+evidence on **the identical real tool trajectories**. Report answer EM, full gold-support citation, original supporting sentence exposure, abstention, search/read/tool calls, total 0.6B+8B model calls, token cost and latency. Include per-question paired uncertainty. **Question gold is evaluator-only after outputs complete**.
5. Success as an engineering module requires **strict citation-and-answer verified success improved over original V4**, not just 8B raw answer quality, while quantifying the incremental 8B inference overhead. This does **not** establish GRPO or RSI weight improvement.
6. If no strict gain appears, stop and record the negative result; do not adapt ranker threshold on this cohort. Keep DEV final-B160 sealed until a genuinely preregistered final protocol.

All new code should carry direct tests for TRAIN64 leakage, source citations, failure handling and model cost. Do not overwrite any experiment artifacts or the frozen cohort manifest.
