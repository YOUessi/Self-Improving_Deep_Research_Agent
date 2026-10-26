# M7-C E-023 — Actual gold-blind two-pass semantic review (2026-10-27)

**Classification: completed real 8B inference and independent source-integrity replay, but no trustworthy novel labels or multi-generation RSI improvement.** This is deliberately a negative result about false bridge reasoning, not a success claim.

## Experiment

Source commit: `dd6527b00462acaa9a944e7c64eed6c4fe377b37` on `feat/m7-rsi-multi-generation`; **156 CPU unit tests passed** on Tang and GitHub CI succeeded. All inputs are the original **TRAIN-only** Qwen3-8B-AWQ grounded Challenger candidates from E-022 (4/8, 3/8, 1/8 quote-valid for cohorts 1, 2, 3). Only the 8 independent-quote-replayed candidates reached this stage. The new `rsi_semantic_triage` code withholds source TRAIN gold, source question answer labels, original Challenger justification, DEV/eval-A/final-B and future Trainer parameters from the semantic reviewing model.

The **same local 8B architecture** served as a **separate role reviewer**, not as an independent trained judge. The verifier ran *two model reviews per candidate*: original document order and reversed document order, `temperature=0`, strict local-only HTTP `127.0.0.1:18986`, with different seeds. The verifier requested a concrete shared bridge entity, two distinct original source quotes and explicit support for each necessary question premise; a separate deterministic checker revalidated document identities, literal quote spans and bridge-string presence in both quotes. **Even two machine approvals never constitute a gold answer or permission to train.**

## Verified experiment counts

| Item | Cohort 1 | Cohort 2 | Cohort 3 | Total |
|---|---:|---:|---:|---:|
| Grounded candidates receiving independent quote replay | 4 | 3 | 1 | **8** |
| Local semantic model-review requests | 8 | 6 | 2 | **16** |
| Flagged unsupported by machine | 3 | 1 | 0 | **4** |
| Invalid/malformed review witness | 1 | 0 | 0 | **1** |
| Two consistent positive model-witness reviews; requires human adjudication | 0 | 2 | 1 | **3** |
| Independently established novel gold labels | 0 | 0 | 0 | **0** |
| New weight updates / RSI generations | 0 | 0 | 0 | **0** |

**Critical false-positive observation:** among **three** apparently consistent positive machine reviewer decisions, human inspection identifies **two clearly unsupported** questions:
- **Batdorf & Bronson ≠ Bronson, Michigan:** quoted company location is **Olympia, Washington**, yet the model twice chose shared string *Bronson* as the bridge and treated **2,349** (the population of Bronson, Michigan) as the answer about Olympia. Exact quoted words and agreement across reversed source ordering **did not prevent a false same-entity match**.
- **Manu Ginóbili / San Antonio Spurs:** both quotes mention the team, but neither supplied quote states the **NBA championship-win** relation demanded by the question. The two-pass reviewer twice accepted the same-team bridge as if it proved that missing event relation.

The remaining candidate concerning the **Desiderata** poem and Max Ehrmann appears textually coherent and should be prioritized for independent semantic adjudication; it has **not** become an approved gold label. The Lee Kyoung-mi / Son Ye-jin question had a complete answer in one document but an unrelated second citation; the structural reviewer output was invalid in both passes. The other four cases were rejected as unsupported.

## Technical finding

Source-document integrity, literal quote matching, extracted answer strings, matching bridge **spellings**, a second source-order inference, and valid generated JSON still **do not establish semantic entailment**. A same-model evaluator is not a truly independent semantic oracle. Claiming that 3/8 are correct would therefore be a serious overstatement. The only training-safe output is `novel_train_labels_approved=0` for all eight.

**Fix priority:** introduce a *negative-only* typed-entity/role consistency guard (Olympia-based versus Michigan population), and a missing-relation evidence guard (championship absent from the quoted source), then rerun *saved* model-review outputs through those CPU guards. These guards can downgrade candidates but must never automatically promote them. For higher recall, add independently annotated two-hop TRAIN novelty labels, adversarial negative benchmarks and an out-of-family verifier. Only after such validation can verified tasks feed real on-policy Solver/Trainer. No sealed final-B160 access.

## Reproducible artifacts

- Training dataset SHA: `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`; original documents SHA: `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`.
- GitHub code: `src/sidra/rsi_semantic_triage.py`, `scripts/run_rsi_semantic_triage.py`, `tests/test_rsi_semantic_triage.py`.
- Tang inputs: `~/experiments/sidra_rsi_m7/challenger/g{1,2,3}_grounded8_qwen3_8b_awq_v1/`. Frozen separate role maps: `~/experiments/sidra_rsi_m7/queues/g{1,2,3}_train24_seed{43,44,45}/`.
- Immutable outputs: `~/experiments/sidra_rsi_m7/semantic_triage/g{1,2,3}_grounded8_goldblind_qwen3_8b_seed{43,44,45}/SEMANTIC_TRIAGE.jsonl` and `SEMANTIC_TRIAGE_SUMMARY.json`; logs `~/experiments/sidra_rsi_m7/logs/g{1,2,3}_semantic_triage_20261027.log`.
- Qwen3-8B-AWQ local server was shut down after the three cohorts. The GPU returned to **349 MiB**, with no other model job interrupted.

**Scientific status:** model-driven negative screening completed; **zero verified novel challenges, zero RSI generations and zero demonstrated multi-step QA improvement**. Branch remains PR draft.
