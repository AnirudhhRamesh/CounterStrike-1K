from __future__ import annotations

import json

import numpy as np
import torch

from cs2_train.src.rollout_archive import RolloutArchiveWriter


def test_rollout_archive_is_complete_hashed_and_round_trips(tmp_path) -> None:
    out_dir = tmp_path / "rollout_archive"
    writer = RolloutArchiveWriter(
        out_dir,
        num_samples=2,
        eval_seeds=[37, 41],
        action_modes=["true", "shuffled"],
        rollout_steps=3,
        height=4,
        width=5,
        num_model_actions=2,
        num_cs2_actions=3,
    )
    ground_truth = torch.linspace(-1, 1, 2 * 3 * 3 * 4 * 5).reshape(2, 3, 3, 4, 5)
    context = torch.zeros(2, 3, 4, 5)
    valid = np.asarray([[True, True, True], [True, True, False]])
    for seed_index in range(2):
        for mode_index in range(2):
            predictions = ground_truth + 0.01 * (seed_index + mode_index)
            model_actions = torch.full((2, 3, 2), float(mode_index))
            cs2_actions = torch.full((2, 3, 3), float(mode_index))
            writer.write_batch(
                seed_index=seed_index,
                mode_index=mode_index,
                sample_positions=[0, 1],
                predictions=predictions,
                ground_truth=ground_truth,
                context_last=context,
                conditioning_actions_model=model_actions,
                conditioning_actions_cs2=cs2_actions,
                valid_steps=valid,
            )

    metadata_path = writer.finalize(
        contract={"checkpoint_step": 50_000, "window_mode": "midpoint"}
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "complete"
    assert metadata["contract"]["eval_seeds"] == [37, 41]
    assert metadata["contract"]["action_modes"] == ["true", "shuffled"]
    assert not list(out_dir.glob(".*.tmp.*"))

    predictions = np.load(out_dir / "predictions_uint8.npy")
    model_actions = np.load(out_dir / "conditioning_actions_model_float32.npy")
    cs2_actions = np.load(out_dir / "conditioning_actions_cs2_float32.npy")
    masks = np.load(out_dir / "valid_steps_bool.npy")
    assert predictions.shape == (2, 2, 2, 3, 4, 5, 3)
    assert model_actions.shape == (2, 2, 3, 2)
    assert cs2_actions.shape == (2, 2, 3, 3)
    assert np.array_equal(
        model_actions[1], np.ones((2, 3, 2), dtype=np.float32)
    )
    assert np.array_equal(cs2_actions[1], np.ones((2, 3, 3), dtype=np.float32))
    assert np.array_equal(masks, valid)
    assert all(
        len(artifact["sha256"]) == 64
        for artifact in metadata["artifacts"].values()
    )


def test_rollout_archive_rejects_incomplete_finalize(tmp_path) -> None:
    writer = RolloutArchiveWriter(
        tmp_path / "rollout_archive",
        num_samples=1,
        eval_seeds=[37],
        action_modes=["true"],
        rollout_steps=1,
        height=2,
        width=2,
        num_model_actions=1,
        num_cs2_actions=1,
    )
    try:
        writer.finalize(contract={})
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("an incomplete archive must not be finalized")


def test_rollout_archive_rejects_changed_shared_data(tmp_path) -> None:
    writer = RolloutArchiveWriter(
        tmp_path / "rollout_archive",
        num_samples=1,
        eval_seeds=[37, 41],
        action_modes=["true"],
        rollout_steps=1,
        height=2,
        width=2,
        num_model_actions=1,
        num_cs2_actions=1,
    )
    common = {
        "mode_index": 0,
        "sample_positions": [0],
        "predictions": torch.zeros(1, 1, 3, 2, 2),
        "context_last": torch.zeros(1, 3, 2, 2),
        "conditioning_actions_model": torch.zeros(1, 1, 1),
        "conditioning_actions_cs2": torch.zeros(1, 1, 1),
        "valid_steps": np.ones((1, 1), dtype=np.bool_),
    }
    writer.write_batch(
        seed_index=0,
        ground_truth=torch.zeros(1, 1, 3, 2, 2),
        **common,
    )
    try:
        writer.write_batch(
            seed_index=1,
            ground_truth=torch.ones(1, 1, 3, 2, 2),
            **common,
        )
    except ValueError as exc:
        assert "ground truth" in str(exc)
    else:
        raise AssertionError("changed ground truth must be rejected")
