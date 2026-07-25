from __future__ import annotations

import json

from cs2_train.scripts.summarize_diamond_rebuttal import (
    ACTION_MODES,
    METRICS,
    load_inline_trajectory,
    summarize_window,
    validate_training_runs,
)


def _rows(training_arm: str) -> list[dict]:
    values = {
        "true": {"true": 1.0, "shuffled": 3.0, "zeros": 4.0},
        "shuffled": {"true": 2.0, "shuffled": 2.5, "zeros": 3.5},
    }[training_arm]
    rows = []
    for dataset_index, round_id in enumerate(("r0", "r1")):
        for mode in ACTION_MODES:
            row = {
                "eval_seed": 37,
                "dataset_index": dataset_index,
                "action_mode": mode,
                "sample_key": f"{round_id}-p0",
                "round_id": round_id,
                "rollout_valid_steps": [True, False],
                "rollout_valid_count": 1,
            }
            row.update({metric: values[mode] for metric in METRICS})
            rows.append(row)
    return rows


def _summary(training_arm: str) -> dict:
    values = {
        "true": {"true": 1.0, "shuffled": 3.0, "zeros": 4.0},
        "shuffled": {"true": 2.0, "shuffled": 2.5, "zeros": 3.5},
    }[training_arm]
    return {
        "means": {
            mode: {metric: values[mode] for metric in METRICS} for mode in ACTION_MODES
        }
    }


def test_cross_arm_summary_reports_sensitivity_and_difference_in_differences() -> None:
    result = summarize_window(
        true_summary=_summary("true"),
        shuffled_summary=_summary("shuffled"),
        true_rows=_rows("true"),
        shuffled_rows=_rows("shuffled"),
        bootstrap_seed=1,
    )
    metric = result["one_step_mse"]
    assert metric["within_checkpoint_action_sensitivity"]["true"]["mean"] == 2.0
    assert metric["within_checkpoint_action_sensitivity"]["shuffled"]["mean"] == 0.5
    assert metric["action_sensitivity_difference_in_differences"]["mean"] == 1.5
    assert metric["training_effect"]["true"]["mean"] == 1.0
    assert result["rollout_target_validity"] == {
        "num_eval_samples": 2,
        "rollout_steps": 2,
        "valid_count_distribution": {"1": 2},
        "valid_target_steps": 2,
        "planned_target_steps": 4,
        "masked_post_alive_target_steps": 2,
        "one_step_targets_all_valid": True,
    }


def test_training_audit_checks_arm_identity_and_checkpoint_reuse(tmp_path) -> None:
    training_commit = "abc123"
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "training_commit.txt").write_text(
        training_commit + "\n", encoding="utf-8"
    )
    shared = {
        "baseline_config": {"schema_version": 1},
        "seed": 28,
        "action_shuffle_seed": 90001,
        "max_steps": 50000,
    }
    summaries = {}
    for arm, checkpoint_hash in (("true", "true-hash"), ("shuffled", "shuffle-hash")):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / "latest.pt").touch()
        (arm_dir / "config.json").write_text(
            json.dumps({**shared, "action_mode": arm}),
            encoding="utf-8",
        )
        for window in ("midpoint", "first-death"):
            summaries[f"{arm}/{window}"] = {
                "checkpoint": str(arm_dir / "latest.pt"),
                "checkpoint_sha256": checkpoint_hash,
            }

    result = validate_training_runs(
        tmp_path,
        summaries,
        expected_training_commit=training_commit,
    )

    assert result["action_mode_by_arm"] == {
        "true": "true",
        "shuffled": "shuffled",
    }
    assert result["checkpoint_sha256_by_arm"]["true"] == "true-hash"


def test_inline_trajectory_requires_every_paired_checkpoint(tmp_path) -> None:
    for arm, offset in (("true", 0.0), ("shuffled", 0.1)):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        rows = []
        for step in (2500, 5000):
            rows.extend(
                [
                    {
                        "kind": "validation",
                        "step": step,
                        "weights": "ema",
                        "true": {"val_mse": 0.2 + offset},
                        "shuffled": {"val_mse": 0.3 + offset},
                        "shuffled_minus_true_mse": 0.1,
                    },
                    {
                        "kind": "rollout",
                        "step": step,
                        "weights": "ema",
                        "true": {"rollout_mse_per_step": [0.2, 0.3]},
                        "shuffled": {"rollout_mse_per_step": [0.25, 0.35]},
                        "true_mse_mean": 0.25,
                        "shuffled_mse_mean": 0.3,
                        "shuffled_minus_true_mse_mean": 0.05,
                    },
                ]
            )
        (arm_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    result = load_inline_trajectory(
        tmp_path,
        expected_step=5000,
        cadence=2500,
    )

    assert [row["step"] for row in result["true"]] == [2500, 5000]
    assert result["true"][0]["one_step"]["shuffled_minus_true_mse"] == 0.1
    assert result["shuffled"][1]["rollout"]["true_mse_per_step"] == [0.2, 0.3]
