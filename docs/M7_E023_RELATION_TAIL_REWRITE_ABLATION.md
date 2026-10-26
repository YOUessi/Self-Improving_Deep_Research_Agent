# M7 E-023 — Question relation-tail second-hop rewrite (frozen NEGATIVE open-pilot result)

**Status: REJECT for promotion.** Source tested on Tang `a5e1777e537fd9d616576186a49ec8cca6a64568` (2026-10-27 Asia/Shanghai); `116 CPU tests passed`. Branch `feat/m7-next-hop-evidence`. No GPU jobs started, no model parameters changed, no hidden DEV/eval-A64/final-B160 used.

## Predeclared gold-blind algorithm and design

Previous E-022 attribution found 37/48 TRAIN Qwen3 rollouts never SEARCH-exposed both required sources; next we investigated a simple fixed **question relation-tail query rewrite** rather than a new verifier threshold.

Algorithm, opted in explicitly:
1. Let existing `plan_next_hop` decide using only question, first genuine READ and exposed frontier, **with the original 6-step/5-tool/Search Top3/READ 3500char budget**. If it chooses to READ an already-visible frontier document, **DO NOT rewrite anything**.
2. Only if it already planned a legal, billed second-hop SEARCH, find the last wh-interrogative word in **raw question tokens before BM25 stopword removal**; take up to 3 nontrivial words *following* that wh word that are not already contained in the proposed bridge query or first READ title; append them to the normal bridge entity query. No extra searches, no corpus inspection, no gold, no answer substitution.
3. Example: question about the Congress of Vienna chaired by `what Austrian statesman` permits query `Congress of Vienna austrian statesman`. This uses only question text and the first READ.
4. Baselines: existing `ProbePolicy(frontier_first)` and `ProbePolicy(forced_bridge)`, all on identical genuine `SearchAgent.search/read/finish` tools and max budget. The only difference is the SEARCH string if/when the frontier-first strategy would search again.
5. **All variants are scripted retrieval diagnostics**. Initial SEARCH/READ, second READ, and FINISH are not Qwen3 model-generated. Labels enter `trace_metrics` only AFTER the agent returns. Output EM / sentence-level citations / semantic entailment / LLM tokens are `null` by design.

### Two TRAIN-only cohorts fixed before pilot

Training source: `train_oracle1024_seed43/converted`; SHA256 questions `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`, documents `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`. 

- **A** seed 43: exact original E-021 TRAIN64 development sample.
- **B** seed 44: predeclared 64 distinct TRAIN questions sampled after excluding all 64 from A; **zero question overlap**.
- Then **one frozen opened DEV pilot32 check** using both unchanged known pilot SHA256 values; results were not used to revise the code. No eval-A64 and absolutely no final-B160 inspection.
- Paired per-question bootstrap, seed 2027, 2,000 resamples and 95% percentile interval.

## Measured retrieval outcome: TRAIN A/B

| Measure | A frontier | A relation-tail | B frontier | B relation-tail |
|---|---:|---:|---:|---:|
| Both gold passages SEARCH-exposed | 37/64 | **38/64** | 33/64 | **35/64** |
| Both gold passages actually READ | 27/64 | **30/64** | 27/64 | **30/64** |
| Mean READ-support recall | .687500 | .703125 | .632812 | .671875 |
| Mean SEARCH calls | 1.62500 | 1.62500 | 1.53125 | 1.53125 |
| Mean billed tools | 3.62500 | 3.62500 | 3.484375 | 3.50000 |
| Proportion with rewritten second SEARCH | 0 | .578125 | 0 | .468750 |

A paired mean READ-support recall delta `+.015625`, 95% CI `[-.015625,.046875]`; B delta `+.039062`, CI `[+.007812,+.070312]`. **Binary full-READ coverage** improvement +3/64 in both cohorts has CI **touching zero**: A `[0,.109375]`, B `[0,.09375]`. These TRAIN results motivated **one** open-pilot diagnostic, not a model promotion. All support metrics are offline labels; policy itself was never given labels.

For reference forced-bridge has both READ 22/64 on A and 19/64 on B; frontier-first alone has 27/64 in both; tail has 30/64 in both, but these are **scripted** outcomes.

## Open pilot32 — regression exposed; NO PROMOTION

The script was **not changed between TRAIN ablation and this fixed pilot check**. Pilot dataset SHA guard: documents `5aabc048b344940ffdce2880a7a57a683d4009777a796fd0602ed683a500cc6c`, questions `94db9bdfff053555b9a8e0da9a850b5bee3de73e2406ec85c63461b5ae033e6e`.

| Metric (N=32; genuine tools, scripted actions) | Frontier-first control | Relation-tail rewrite |
| --- | ---: | ---: |
| Both support docs SEARCH-exposed | **22/32** | **20/32** |
| Both support docs READ | **16/32** | **16/32** |
| Mean READ-support recall | .671875 | .687500 |
| Mean SEARCH calls | 1.625 | 1.625 |
| Mean billed tool calls | **3.59375** | **3.62500** |
| Proportion with rewritten second SEARCH | 0 | .5625 |

The predeclared pilot comparison shows **−2/32** fully exposed gold passage pairs (delta `−.0625`, paired bootstrap interval `[−.15625,0]`), **ZERO gain** in full-READ coverage (delta `0`, interval `[−.09375,.09375]`), and **+0.03125** average tool calls (interval `[0,.09375]`). Mean READ-support recall +.015625 with interval `[−.046875,.078125]`, i.e. no convincing overall evidence gain.

**Decision:** preserve the candidate and TRAIN success as a negative generalization example; **do NOT** enable this rewrite in the default Search Agent, do not tune it repeatedly on the pilot and do not touch independent eval-A64 or final-B160. The chosen E-021 frontier-first also remains experimental rather than a claimed learned-LLM gain. The *actual* Qwen3 model must separately be tested with strictly matched decoding, real tool turns, correct answers and sentence evidence before any promotion.

## Reproducibility and bug log

- Source: `src/sidra/bridge_rewrite.py`, `scripts/m7_relation_tail_probe.py`, `tests/test_bridge_rewrite.py`.
- Tang exact source snapshot: `~/experiments/sidra_m7_evidence_run2/source_a5e1777e537fd9d616576186a49ec8cca6a64568/`.
- Local immutable output JSON (TRAIN per-question metrics; DEV aggregate-only):
  `~/experiments/sidra_m7_evidence_run2/results/rewrite_train64x2_v1/summary.json`
  and `.../results/rewrite_pilot32_v1_frozen/summary.json`.
- Reproduce:
  ```bash
  PYTHONPATH=src:. python3 scripts/m7_relation_tail_probe.py --mode train \
    --dataset ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
    --out <UNUSED_DIR> --n 64

  PYTHONPATH=src:. python3 scripts/m7_relation_tail_probe.py --mode pilot \
    --dataset ~/experiments/sidra_m4/data/frozen_seed42_32p_224h/pilot/converted \
    --out <UNUSED_DIR>
  ```
- Initial **114 passed / 2 failed** tests on snapshot `ebffda4` exposed two real issues BEFORE any dataset ablation:
  - **Wh-question lost to BM25 stopword removal**: `tokenize()` removes `what`; relation suffix must be parsed from raw lowercase word tokens before BM25 tokenization. Fixed in `c5546f3`.
  - **Diagnostic metadata affected action selection**: an injected `eval_only` event after a READ changed the branch condition, thus a policy could potentially react to forbidden evaluator metadata. Fixed by filtering *only* legitimate SEARCH/READ observations before both branching and delegation in `a5e1777`.
  - Verified final **116/116 CPU tests passed** on Tang, including gold-perturbation, disjoint cohort, tool budget and frozen pilot SHA guard.
- Future candidate designs should be selected on **independent TRAIN** and frozen before limited DEV pilot checks. This attempt's DEV pilot is now a **negative recorded result**; further changes need a new prospective TRAIN hypothesis with strict multiple-try disclosure, not post-hoc pilot tuning.
