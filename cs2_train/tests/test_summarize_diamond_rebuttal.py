from __future__ import annotations

from cs2_train.scripts.summarize_diamond_rebuttal import (
    ACTION_MODES,
    METRICS,
    summarize_window,
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
