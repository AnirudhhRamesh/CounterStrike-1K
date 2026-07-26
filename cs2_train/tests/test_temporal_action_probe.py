from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from cs2_train.src.temporal_action_probe import (
    ACTION_LABEL_NAMES,
    DINOV2_EXPECTED_WEIGHT_SHA256,
    DINOV2_REPOSITORY,
    TemporalActionProbe,
    actions_to_labels,
    binary_average_precision_tie_aware,
    split_probe_segments,
)


def test_temporal_action_labels_use_presence_and_signed_mouse_sum() -> None:
    actions = torch.zeros(2, 4, 14)
    actions[0, 1, 0] = 1
    actions[0, :, 12] = torch.tensor([-0.4, -0.4, -0.4, 0.0])
    actions[0, :, 13] = torch.tensor([0.4, 0.4, 0.4, 0.0])
    actions[1, 2, 7] = 1
    labels = actions_to_labels(actions)
    assert labels.shape == (2, len(ACTION_LABEL_NAMES))
    assert labels[0, 0].item() == 1
    assert labels[0, 10].item() == 1  # pitch negative
    assert labels[0, 13].item() == 1  # yaw positive
    assert labels[1, 7].item() == 1
    assert labels[1, 10:].sum().item() == 0


def test_probe_segments_share_boundary_frame_and_not_transitions() -> None:
    features = torch.arange(2 * 9 * 3).reshape(2, 9, 3).float()
    actions = torch.zeros(2, 8, 14)
    actions[:, 0, 0] = 1
    actions[:, 4, 7] = 1
    segments, labels = split_probe_segments(features, actions)
    assert segments.shape == (2, 2, 5, 3)
    assert torch.equal(segments[:, 0, -1], features[:, 4])
    assert torch.equal(segments[:, 1, 0], features[:, 4])
    assert labels[:, 0, 0].eq(1).all()
    assert labels[:, 1, 7].eq(1).all()
    assert labels[:, 0, 7].eq(0).all()


def test_temporal_probe_shape_and_parameter_budget() -> None:
    model = TemporalActionProbe(
        input_dim=16,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )
    logits = model(torch.randn(3, 5, 16))
    assert logits.shape == (3, len(ACTION_LABEL_NAMES))

    release_model = TemporalActionProbe()
    parameters = sum(parameter.numel() for parameter in release_model.parameters())
    assert 4_000_000 < parameters < 5_000_000


def test_average_precision_groups_ties_and_is_row_order_invariant() -> None:
    labels = torch.tensor([1, 0, 1, 0]).numpy()
    constant = torch.zeros(4).numpy()
    assert binary_average_precision_tie_aware(labels, constant) == 0.5
    assert binary_average_precision_tie_aware(labels[::-1], constant[::-1]) == 0.5

    scores = torch.tensor([0.9, 0.8, 0.8, 0.1]).numpy()
    expected = binary_average_precision_tie_aware(labels, scores)
    permutation = [2, 0, 3, 1]
    assert (
        binary_average_precision_tie_aware(
            labels[permutation],
            scores[permutation],
        )
        == expected
    )


def test_frozen_arr_config_and_launcher_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "cs2_train/configs/temporal_arr_cs1k_dust2_v1.json"
    launcher = root / "cs2_train/scripts/run_temporal_action_probe_dust2_v1.sh"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["status"] == "frozen-before-probe-training"
    assert (
        config["dataset"]["manifest_sha256"]
        == "33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e"
    )
    assert config["dataset"]["train_windows"] == 100_000
    assert config["dataset"]["validation_windows"] == 10_000
    assert config["dataset"]["target_fps"] == 8
    assert config["backbone"]["repository"] == DINOV2_REPOSITORY
    assert config["backbone"]["weights_sha256"] == DINOV2_EXPECTED_WEIGHT_SHA256
    assert config["labels"]["buttons"] == list(ACTION_LABEL_NAMES[:10])
    assert config["labels"]["mouse"] == list(ACTION_LABEL_NAMES[10:])
    assert config["endpoint"]["average_precision"] == (
        "threshold-based and exact-tie-aware"
    )
    subprocess.run(["bash", "-n", str(launcher)], check=True)
