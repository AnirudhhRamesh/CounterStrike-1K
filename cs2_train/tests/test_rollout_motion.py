from __future__ import annotations

import numpy as np
import torch

from cs2_train.src.evaluate_rollout_motion import (
    central_world_crop,
    flow_agreement_per_step,
    masked_sample_means,
    paired_round_bootstrap,
)


def test_identical_flow_has_zero_dynamics_error() -> None:
    flow = torch.randn(2, 3, 2, 8, 12)
    metrics = flow_agreement_per_step(
        flow,
        flow,
        crop=central_world_crop(
            8,
            12,
            top_fraction=0.0,
            bottom_fraction=1.0,
            side_fraction=0.0,
        ),
    )
    assert all(torch.equal(value, torch.zeros_like(value)) for value in metrics.values())


def test_flow_metrics_apply_alive_mask_and_normalize_by_diagonal() -> None:
    target = torch.zeros(1, 2, 2, 6, 8)
    predicted = target.clone()
    predicted[:, 0, 0] = 1.0
    predicted[:, 1, 0] = 100.0
    metrics = flow_agreement_per_step(
        predicted,
        target,
        crop=(slice(0, 6), slice(0, 8)),
    )
    per_step, means = masked_sample_means(
        metrics["flow_epe_normalized"],
        np.asarray([[True, False]]),
    )
    assert per_step[0][1] is None
    assert abs(means[0] - 0.1) < 1e-6  # 1 px / sqrt(6^2 + 8^2)


def test_paired_motion_delta_is_positive_when_true_is_better() -> None:
    rows = []
    for sample_position, round_id in enumerate(("r0", "r1")):
        for action_mode, value in (("true", 0.1), ("shuffled", 0.3)):
            rows.append(
                {
                    "eval_seed": 37,
                    "sample_position": sample_position,
                    "round_id": round_id,
                    "action_mode": action_mode,
                    "flow_epe_normalized_mean": value,
                }
            )
    summary = paired_round_bootstrap(
        rows,
        metric="flow_epe_normalized_mean",
        bootstrap_seed=7,
        n_bootstrap=100,
    )
    assert abs(summary["shuffled"]["mean_delta"] - 0.2) < 1e-9
    assert summary["shuffled"]["ci95"][0] > 0
