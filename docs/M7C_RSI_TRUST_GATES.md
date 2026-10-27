# M7-C: RSI challenger / solver / verifier trust and generation ledger

**State:** prerequisite governance scaffold implemented; NO actual multi-generation RSI algorithm or model improvement shown.

The code in `src/sidra/rsi_ledger.py` prepares auditable role boundaries for a future Challenger → Solver → Verifier → Trainer process. It does **not** claim the system can yet invent reliably solvable new tasks or learn from them.

## Trust domains / data rights

| Role | May read | May NOT access | Required proof |
| --- | --- | --- | --- |
| Challenger | Original HotpotQA TRAIN question text, recorded source excerpts, current generation metadata | DEV labels, sealed final-B, evaluator answer keys | Proposed TRAIN source ID and normalized challenge SHA; deduplicated record |
| Solver | Frozen candidate challenge question, ordinary bounded SearchAgent tools | Source gold label, verifier grader, DEV evaluator | Saved model actions, read-before-cite trace |
| Verifier | Original TRAIN ground truth and model trace | DEV/TEST gold labels as training reward | Independent offline train-only answer/citation check; reject unsupported new challenger tasks |
| Trainer | TRAIN-approved source trajectories, reward and adapter checkpoint | DEV/TEST answer labels or negative examples derived from sealed data | Actual gradient update, new adapter SHA, grouped objective |
| Evaluator | Frozen DEV QA with original supporting facts | Sending gold back to Challenger/Trainer | Predeclared same-dataset before/after results with cost and uncertainty |

Key research issue: a *novel Challenger question* is NOT entitled to inherit the original training question's gold answer. Until independent TRAIN-only verification or valid re-labelling is implemented, the Verifier marks it `unlabelled_challenge`, not a training positive. This protects the RSI loop from self-generated false labels.

## Implemented proof barriers

- `freeze_train_challenges()`: checks TRAIN-only origins, rejects extra gold-containing proposal fields, deduplicates normalized question text, writes immutable challenger/solver inputs in separate files.
- `verify_train_traces()`: offline reviewer permits train reward only for question-equivalent established training tasks; genuinely new questions need independent labels and must not inherit source truth.
- `record_generation()`: different old/new SHA-256 and frozen DEV comparison of at least 32 questions required for one-generation promotion. Negative results retained. **A single successful generation still explicitly marks `multi_generation_RSI_completed=false`.**
- `tests/test_rsi_ledger.py`: checks role redaction, no train/dev overlap, novel-unlabelled refusal, checkpoint change and nonpositive generation rejection.

## What remains to reach the user's actual objective

1. Implement a *real model-driven* Challenger using TRAIN-only prompts and retrieved context; verify diversity/difficulty across seeds and detect near-duplicates.
2. Execute real Solver search/read/finish on its candidates; require correct labels supported at original sentence level, not self-attested model outputs.
3. Integrate proper on-policy Agentic GRPO updates and checkpoint versioning; run >1 successive generation with new online TRAIN rollouts each generation.
4. Pre-register independent DEV evaluation and keep sealed final-B160 untouched until a final protocol. Report direct evidence of actual answer and citation gains across multiple seeds and model generations.
5. Incorporate adversarial negative retrieval and untrusted-document prompt injection tests across role boundaries.

Source-of-truth branch: `feat/m7-rsi-multi-generation`. CPU CI tests must pass before merging. This branch is intentionally independent of parallel M7 on-policy GRPO and search/Verifier work.
