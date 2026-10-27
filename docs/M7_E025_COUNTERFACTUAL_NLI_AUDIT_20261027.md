# M7 E-025 — Counterfactual citation-necessity NLI falsification (TRAIN-only)

**2026-10-27, CPU executed on Tang.** GitHub target branch `feat/m7-next-hop-evidence`. Executed source commit `8af0701e9565a1300607e508383bfbee29bf8e54`. Tang CPU verified **149/149 unit tests passed** at that SHA; CI result tracked separately. No GPU model inference/training was launched, other GPU jobs were neither blocked nor interrupted. Unmodified M6 real Qwen3 tool-rollout traces were replayed, with the exact original SearchAgent tool budgets and observations. No further SEARCH or READ action was fabricated.

**Research question:** Does an NLI model's high probability of entailment reflect dependence on **both** actually read and cited pieces of evidence? Our previous E-024 source-only title-link screen reduced 8 confident predictions to 2, but was not proof of semantic correctness, and still passed one correct answer lacking full original sentence evidence.

## Fixed opt-in algorithm (before reading outcomes)

After a completed, legal `finish` with authentic read-before-cite and baseline MNLI `p_entail>=0.85`:

1. Build the existing question + **MODEL ANSWER** hypothesis. Do not use original HotpotQA answer to build the claim. Existing M7 source-only checks reject malformed, unread, duplicated or insufficient citations. Only actual matched READ observations form the premise; SEARCH snippets and hidden corpus are forbidden.
2. Freeze the same deterministic sentence-window selection for every cited document; **abstain** if combining citations would change window content by context reallocation or global character truncation. Check actual CPU model tokenizer length of all full/ablated pairs before it can silently truncate.
3. For each citation `c`, recompute NLI after **removing only c's actual READ window**. Keep other windows and claim unchanged. This is an evidence-removal experiment, not a counterfactual world intervention and not a gold-aware evaluator.
4. Mark `counterfactual_necessary_model_prediction` **only if every** removed citation produces `p_entail(without c)<=0.65` and `p_full-p_without>=0.20`. Otherwise reject the necessity claim. Never call this semantic proof.
5. Independently record original TRAIN gold answer EM and **original HotpotQA (title,sentence_index) sentence coverage inside actually cited READ text** only **after** the NLI screening. These remain entirely outside the scorer/policy trust domain.

This additional test is compared on exactly the same 80 real model episodes to (a) baseline raw MNLI and (b) prior observed cross-document title-link gate. It is a **post-hoc diagnostic and opt-in CPU experiment**; no default Verifier change.

## Source manifests, data isolation

- TRAIN-only converted questions SHA256 `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`.
- TRAIN-only converted documents SHA256 `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`.
- Original **TRAIN** HotpotQA sentence-label file SHA256 `83f03106f12d2c22f36a6957527889fd80f3d13100b03255fc42028ad218f417`, all 1,024 converted TRAIN IDs validated before NLI runs.
- Real rollouts: M6 Qwen3-0.6B `real_train_rollouts12x4_stop`, 48 episodes SHA256 `1f1129d8f94bc7a212590a63b14299e917a4f5e6b13b8ffab4a4079ebd492ffb`; `train_rollout_io8x4`, 32 episodes SHA256 `5af71a10d57b15e91ca4a400126ce3c667c9784793caa4dd7f13709bacce4aad`.
- Important: two previous groups have *overlapping question IDs*, hence **80 episodes but only 12 distinct TRAIN question clusters**. Do NOT bootstrap episodes independently or pretend there are 20 independent questions.
- CPU model: cached `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` with `DebertaV2ForSequenceClassification`, checkpoint SHA256 `65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e`, max pair tokens384, no remote download.
- No DEV gold, no eval-A64, no sealed final-B160, no Qwen3 weight update.

## Actual 80-episode observations — **negative promotion finding**

| Outcome on the **same 80 real episodes** | Raw confident MNLI | Prior title-link gate | New LOCO necessity gate | Intersection: title-link AND LOCO |
| --- | ---: | ---: | ---: | ---: |
| Accepted model-predicted entailment | **8** | **2** | **2** | **1** |
| Accepted even though answer EM **wrong** | **6** | **0** | **1** | **0** |
| Accepted with **missing original cited supporting sentence(s)** | **7** | **1** | **1** | **0** |

NLI scored only **21/80** episodes; other 59 were abstained, illegal, lacked source lexical anchor or unsupported question-to-claim grammar. Counterfactual test ran on **8 eligible confident entailment** episodes. LOCO outcomes: **2 passed necessity**, **6 revealed a nonessential citation**; **16 additional CPU NLI calls**, average ~0.2 calls and 14.6ms per original episode *for the counterfactual stage only*. This is not total pipeline end-to-end latency.

The true-positive proxy is answer-EM plus original sentence presence, **NOT independently annotated entailment truth**:
- TRAIN question `5ab5ae8d5542992aa134a383` / episode2: answer `Tayto`, EM correct and all original facts in cited READ. Baseline p=.87943. Removing either citation reduces p to .02384 or .00332. Both screens pass (1 positive).
- TRAIN question `5a8369715542996488c2e440` / episode1: answer `1962`, **EM wrong** and required original sentence absent. Yet raw baseline p=.96385 and removal of citations yields .00175/.21256: **LOCO passes a wrong answer**, while title-link rejects. This is the decisive negative counterexample: **both documents contributing to a confident wrong inference does not establish factual correctness**.
- TRAIN question `5a89de0a554299669944a5eb` / episode3: answer `Jean-Baptiste Lully`, EM correct but original support sentences not all cited. Title-link passes; removing either citation makes MNLI even MORE confident (.98584/.99666 > baseline .97760), so LOCO correctly detects lack of citation necessity. Neither single gate is a verified semantic adjudicator.
- Repeated TRAIN `MacVenture` episodes produced confident p=.99555 but **one citation could be removed with p=.99402**; LOCO blocks them. Several repeats refer to the same question, not independent examples.

### Correct question-cluster uncertainty (5,000 draws, fixed seed2027)

- LOCO minus title-link acceptance rate: **0/80 difference**, clustered 95% CI **[-0.039474, +0.037500]**.
- LOCO minus title-link wrong-answer accepted: **+1/80**, clustered 95% CI **[0, +0.041667]**; not an independent significant comparison.
- LOCO acceptance rate: **2/80 = 0.025**, cluster 95% CI **[0,0.0625]**.
- Intersection acceptance rate: **1/80 = 0.0125**, cluster 95% CI **[0,0.0375]**.
- All original-gold EM and original-sentence coverage metrics are unchanged by construction, as this is a *post-hoc verifier ablation*, not new model inference.

**Decision:** **DO NOT PROMOTE LOCO AS A RELIABLE SEMANTIC VERIFIER.** Its important value is a falsification/diagnostic tool identifying confidently irrelevant citations; one newly confirmed wrong-answer false approval shows necessity is not sufficient. The conjunction's 1/80 acceptance is too sparse for operational use or strong claims. No repeated Pilot32 tuning and no eval-A64/final-B160 release. A human-labelled TRAIN-derived entailment validation set with both positive and hard contradiction/near-miss cases is required before calibrating threshold or running a prospective fixed-policy Qwen3 QA experiment.

## Reproduce on Tang (CPU only, no overwrite)

Source SHA above, then from its extracted snapshot:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
PYTHONPATH=src:. ~/projects/CodeRL-Lab/.venv/bin/python scripts/m7_counterfactual_nli_audit.py \
 --dataset ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
 --original-train ~/experiments/sidra_m4/data/train_oracle1024_seed43/original_train_subset.json \
 --rollouts ~/experiments/sidra_m6/results/real_train_rollouts12x4_stop/rollouts_train.jsonl \
 --rollouts ~/experiments/sidra_m6/results/train_rollout_io8x4/rollouts_train.jsonl \
 --model ~/ .cache/huggingface/hub/models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli/snapshots/<SNAPSHOT> \
 --out <NEW_NONEXISTENT_OUTPUT_DIRECTORY> --bootstrap 5000
```

**NB:** replace the illustrative model path placeholder with the actual cached snapshot path **without whitespace** (`~/.cache/...`). The tested absolute snapshot on Tang is `/home/you/.cache/huggingface/hub/models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli/snapshots/8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`.

Immutable experimental output: `~/experiments/sidra_m7_evidence_run4/results/train80_mdeberta_loco_20261027/{summary.json,train_only_per_episode.jsonl}`; raw console log `~/experiments/sidra_m7_evidence_run4/logs/train80_mdeberta_loco_20261027.log`. Files include episode IDs and all per-citation ablated model probabilities for auditing. **No output overwritten.**

## Next evidence gates

1. Build an independent TRAIN-question-group, manually or strongly adversarially adjudicated **semantic entailment truth** benchmark, with plausible wrong bridge joins, flipped relations, negation scope, and required joint support. Gold answers and sentence indices must remain evaluator-only. Threshold selection only on TRAIN and no DEV peeking.
2. Explicitly analyze read-before-cite legal-but-irrelevant documents, NLI window truncation and model lexical shortcuts; compare conditional false-approval vs false-reject separately with independent labels.
3. Only after a frozen calibrated verifier and an actual model-policy deployment test on *one* already-open Pilot32 with unchanged decoding and budgets should independent **eval-A64** be considered. Final-B160 stays sealed.
