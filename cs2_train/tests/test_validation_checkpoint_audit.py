from __future__ import annotations

from cs2_train.scripts.run_diamond_validation_checkpoint_audit import (
    action_sensitivity_by_key,
    endpoint_change,
)


def _rows(true_mse: float, shuffled_mse: float) -> list[dict]:
    rows = []
    for dataset_index, round_id in enumerate(("r0", "r1")):
        for action_mode, value in (
            ("true", true_mse),
            ("shuffled", shuffled_mse),
            ("zeros", shuffled_mse + 1),
        ):
            rows.append(
                {
                    "eval_seed": 37,
                    "dataset_index": dataset_index,
                    "sample_key": f"{round_id}-p0",
                    "round_id": round_id,
                    "action_mode": action_mode,
                    "one_step_mse": value,
                }
            )
    return rows


def test_checkpoint_change_is_paired_difference_in_sensitivity() -> None:
    old = _rows(1.0, 1.5)
    new = _rows(0.8, 1.8)
    result = endpoint_change(
        old,
        new,
        metric="one_step_mse",
        bootstrap_seed=1,
    )
    assert result["mean"] == 0.5
    assert result["trend"] == "increasing"


def test_action_sensitivity_keys_retain_round_clusters() -> None:
    result = action_sensitivity_by_key(
        _rows(1.0, 1.25),
        "one_step_mse",
    )
    assert result[(37, 0)] == ("r0", 0.25)
    assert result[(37, 1)] == ("r1", 0.25)
