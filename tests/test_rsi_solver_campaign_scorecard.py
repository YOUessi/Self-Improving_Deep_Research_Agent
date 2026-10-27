"""E-028 refuses to misrepresent real constrained-decoder scouts as GRPO."""
from copy import deepcopy

import pytest

from sidra.rsi_solver_campaign_scorecard import (
    combine_verified_scouts, audit_cross_cohort_solver_campaign,
)


def pair(cohort: int, seed: int, *, strict: int = 0, correct: int = 1):
    run = {
        "stage": "actual_model_solver_tool_rollouts_NOT_training",
        "cohort": cohort, "seed": seed, "model_id": "frozen_Qwen3_06B_stop",
        "model_checkpoint_sha256_user_supplied": "a" * 64,
        "decoder": "vLLM_dynamic_JSON_schema_observed_docIDs",
        "temperature": 0.75, "top_p": 0.9,
        "config": {"max_steps": 6, "max_tool_calls": 5, "search_top_k": 3},
        "n_episodes": 2, "model_calls": 6,
        "prompt_tokens": 400, "completion_tokens": 50,
        "traces_sha256": f"{cohort:064x}",
        "model_action_io_sha256": f"{cohort + 20:064x}",
        "TRAIN_gold_loaded_or_graded_by_solver": False,
        "on_policy_training_distribution_proven": False,
        "optimizer_updates": 0, "multi_generation_RSI_proven": False,
    }
    audit = {
        "n": 2, "tool_replay_valid": 2,
        "gold_read_only_in_separate_verifier": True,
        "model_provenance_independently_attested": False,
        "training_updates_this_run": 0, "optimizer_eligible": False,
        "multi_generation_RSI_proven": False,
        "solver_traces_sha256": run["traces_sha256"],
        "model_actions_sha256": run["model_action_io_sha256"],
        "model_action_io_replay": {
            "verified_episode_IO_count": 2, "verified_response_count": 6,
            "all_model_contexts_and_seeds_match_tool_replay": True,
        },
        "difficulty_tier_aggregate": {
            "direct": {"n": 1, "strict": strict, "answer_correct": correct},
            "bridge": {"n": 1, "strict": 0, "answer_correct": 0},
            "hard": {"n": 0, "strict": 0, "answer_correct": 0},
        },
        "strict_original_TRAIN_successes": strict,
        "legal_finished_answers": 1,
        "nonerror_terminations": 2,
        "mean_gold_support_read_recall": 0.5,
    }
    return run, audit


def good():
    return [pair(i, 42 + i) for i in (1, 2, 3)]


def test_verified_cross_cohort_reducer_keeps_real_training_blocked():
    r = combine_verified_scouts(good(), planned_cohorts=3)
    assert r["actual_distinct_original_TRAIN_questions"] == 6
    assert r["answer_correct_count"] == 3
    assert r["strict_answer_plus_gold_passage_evidence_count"] == 0
    assert r["answered_correctly_but_not_strictly_verified"] == 3
    assert r["unique_solver_sampling_seeds"] == [43, 44, 45]
    assert r["mean_support_read_recall"] == 0.5
    assert r["model_calls"] == 18
    assert r["trainer_policy"] if "trainer_policy" in r else not r["eligible_for_online_GRPO_training_from_these_files"]
    assert r["real_optimizer_updates"] == r["real_model_generations"] == 0


@pytest.mark.parametrize("mutation", [
    "missing_cohort", "duplicate_cohort", "seed_reuse",
    "model_mismatch", "decoder_mismatch", "altered_trace_sha",
    "invalid_checkpoint", "fake_optimizer", "gold_in_solver",
    "fake_verified_io", "unverified_model_origin", "incorrect_n",
    "invalid_tier", "strict_gt_correct", "duplicate_trace",
    "fake_training_label",
])
def test_tampered_or_misleading_runs_fail_closed(mutation):
    items = good()
    if mutation == "missing_cohort":
        items.pop()
    elif mutation == "duplicate_cohort":
        items[1][0]["cohort"] = 1
    elif mutation == "seed_reuse":
        items[2][0]["seed"] = 43
    elif mutation == "model_mismatch":
        items[1][0]["model_id"] = "injected-model"
    elif mutation == "decoder_mismatch":
        items[2][0]["decoder"] = "different_decoder"
    elif mutation == "altered_trace_sha":
        items[1][1]["solver_traces_sha256"] = "d" * 64
    elif mutation == "invalid_checkpoint":
        items[0][0]["model_checkpoint_sha256_user_supplied"] = "123"
    elif mutation == "fake_optimizer":
        items[1][0]["optimizer_updates"] = 1
    elif mutation == "gold_in_solver":
        items[0][0]["TRAIN_gold_loaded_or_graded_by_solver"] = True
    elif mutation == "fake_verified_io":
        items[2][1]["model_action_io_replay"]["verified_response_count"] = 0
    elif mutation == "unverified_model_origin":
        items[0][1]["model_provenance_independently_attested"] = True
    elif mutation == "incorrect_n":
        items[0][1]["n"] = 3
    elif mutation == "invalid_tier":
        items[2][1]["difficulty_tier_aggregate"].pop("bridge")
    elif mutation == "strict_gt_correct":
        items[1][1]["difficulty_tier_aggregate"]["direct"]["strict"] = 2
    elif mutation == "duplicate_trace":
        items[1][0]["traces_sha256"] = items[0][0]["traces_sha256"]
        items[1][0]["model_action_io_sha256"] = items[0][0]["model_action_io_sha256"]
        items[1][1]["solver_traces_sha256"] = items[1][0]["traces_sha256"]
        items[1][1]["model_actions_sha256"] = items[1][0]["model_action_io_sha256"]
    elif mutation == "fake_training_label":
        items[1][1]["optimizer_eligible"] = True
    with pytest.raises(ValueError):
        combine_verified_scouts(items, planned_cohorts=3)


def test_refuses_sealed_finalB_path_before_any_file_open(tmp_path):
    with pytest.raises(ValueError, match="refuse heldout/final-B"):
        audit_cross_cohort_solver_campaign(
            train_data=tmp_path / "sealed" / "final-B160",
            campaign=tmp_path / "campaign",
            packs=[tmp_path / "pack1", tmp_path / "pack2"],
            runs=[tmp_path / "run1", tmp_path / "run2"],
            source_sha="a" * 40, out=tmp_path / "out.json",
        )


def test_refuses_reusing_existing_output_before_reading_data(tmp_path):
    out = tmp_path / "audit.json"
    out.write_text("KEEP")
    with pytest.raises(FileExistsError):
        audit_cross_cohort_solver_campaign(
            train_data=tmp_path / "train", campaign=tmp_path / "campaign",
            packs=[tmp_path / "p1", tmp_path / "p2"],
            runs=[tmp_path / "r1", tmp_path / "r2"],
            source_sha="a" * 40, out=out,
        )
    assert out.read_text() == "KEEP"
