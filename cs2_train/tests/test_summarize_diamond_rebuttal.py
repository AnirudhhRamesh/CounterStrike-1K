from __future__ import annotations

import json

import pytest

from cs2_train.scripts.summarize_diamond_rebuttal import (
    ACTION_MODES,
    CROSS_WINDOW_IDENTICAL_FIELDS,
    METRICS,
    load_inline_trajectory,
    main as summarize_main,
    summarize_window,
    validate_training_runs,
    validate_window_contracts,
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


def _window_contract(*, sample_plan: str, action_plan: str) -> dict:
    shared = {
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "split": "test",
        "map_slug": "dust2",
        "target_fps": 8,
        "resize": [36, 64],
        "rollout_steps": 8,
        "rollout_target_masking": "source_frame < alive_end_frame",
        "num_denoising_steps": 1,
        "s_cond": 0.005,
        "eval_seeds": [37, 41, 43],
        "action_modes": ["true", "shuffled", "zeros"],
        "num_eval_samples": 690,
        "num_rounds": 69,
    }
    assert set(shared) == set(CROSS_WINDOW_IDENTICAL_FIELDS)
    return {
        **shared,
        "sample_plan_sha256": sample_plan,
        "action_plan_sha256": action_plan,
    }


def test_window_contracts_allow_only_window_specific_plans_to_differ() -> None:
    contracts = {
        "midpoint": _window_contract(
            sample_plan="midpoint-samples",
            action_plan="midpoint-actions",
        ),
        "first-death": _window_contract(
            sample_plan="death-samples",
            action_plan="death-actions",
        ),
    }

    shared = validate_window_contracts(contracts)

    assert shared["manifest_sha256"] == "manifest"
    assert "sample_plan_sha256" not in shared
    assert "action_plan_sha256" not in shared


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("config_sha256", "other-config"),
        ("eval_seeds", [37, 41]),
        ("num_eval_samples", 689),
        ("num_rounds", 68),
        ("rollout_target_masking", "none"),
    ],
)
def test_window_contracts_reject_cross_window_drift(field, different) -> None:
    contracts = {
        "midpoint": _window_contract(
            sample_plan="midpoint-samples",
            action_plan="midpoint-actions",
        ),
        "first-death": _window_contract(
            sample_plan="death-samples",
            action_plan="death-actions",
        ),
    }
    contracts["first-death"][field] = different

    with pytest.raises(ValueError, match=f"window modes differ on {field}"):
        validate_window_contracts(contracts)


def test_summarizer_cli_retains_shared_and_window_specific_contracts(
    tmp_path,
    monkeypatch,
) -> None:
    training_commit = "abc123"
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "training_commit.txt").write_text(
        training_commit + "\n",
        encoding="utf-8",
    )
    window_contracts = {
        "midpoint": {
            **_window_contract(
                sample_plan="midpoint-samples",
                action_plan="midpoint-actions",
            ),
            "eval_seeds": [37],
            "num_eval_samples": 2,
            "num_rounds": 2,
        },
        "first-death": {
            **_window_contract(
                sample_plan="death-samples",
                action_plan="death-actions",
            ),
            "eval_seeds": [37],
            "num_eval_samples": 2,
            "num_rounds": 2,
        },
    }
    baseline_config = {"schema_version": 1, "purpose": "test"}
    for arm, checkpoint_hash in (("true", "true-hash"), ("shuffled", "shuffle-hash")):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        checkpoint = arm_dir / "latest.pt"
        checkpoint.touch()
        (arm_dir / "config.json").write_text(
            json.dumps(
                {
                    "action_mode": arm,
                    "baseline_config": baseline_config,
                    "val_every": 2500,
                }
            ),
            encoding="utf-8",
        )
        inline_rows = []
        for step in (2500, 5000):
            inline_rows.extend(
                [
                    {
                        "kind": "validation",
                        "step": step,
                        "weights": "ema",
                        "true": {"val_mse": 0.2},
                        "shuffled": {"val_mse": 0.3},
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
            "".join(json.dumps(row) + "\n" for row in inline_rows),
            encoding="utf-8",
        )
        for window_mode, contract in window_contracts.items():
            evaluation = arm_dir / "evaluation" / window_mode
            evaluation.mkdir(parents=True)
            summary = {
                **contract,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_step": 5000,
                "means": _summary(arm)["means"],
            }
            (evaluation / "summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            (evaluation / "per_sample_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in _rows(arm)),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_diamond_rebuttal",
            "--run-root",
            str(tmp_path),
            "--expected-step",
            "5000",
            "--expected-samples",
            "2",
            "--expected-training-commit",
            training_commit,
        ],
    )
    summarize_main()

    output = json.loads(
        (tmp_path / "evaluation" / "rebuttal_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert output["schema_version"] == 2
    assert output["contract"]["manifest_sha256"] == "manifest"
    assert "sample_plan_sha256" not in output["contract"]
    assert (
        output["contract_by_window"]["midpoint"]["sample_plan_sha256"]
        == "midpoint-samples"
    )
    assert (
        output["contract_by_window"]["first-death"]["sample_plan_sha256"]
        == "death-samples"
    )
    markdown = (tmp_path / "evaluation" / "rebuttal_summary.md").read_text(
        encoding="utf-8"
    )
    assert "Test samples: 2 POV rows" in markdown
