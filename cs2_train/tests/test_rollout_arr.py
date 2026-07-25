from __future__ import annotations

import numpy as np

from cs2_train.src.evaluate_rollout_arr import (
    segment_validity,
    summarize_arr,
    weighted_cluster_bootstrap_ap,
)
from cs2_train.src.temporal_action_probe import ACTION_LABEL_NAMES


def _weighted_ap(labels, scores, clusters, counts) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ranked = labels[order]
    weights = counts[clusters[order]]
    positive_weights = ranked * weights
    cumulative_weight = np.cumsum(weights)
    cumulative_positive = np.cumsum(positive_weights)
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    positives_at_threshold = cumulative_positive[ends]
    group_positive = np.diff(np.r_[0, positives_at_threshold])
    precision = np.divide(
        positives_at_threshold,
        cumulative_weight[ends],
        out=np.zeros_like(positives_at_threshold, dtype=np.float64),
        where=cumulative_weight[ends] > 0,
    )
    return float((precision * group_positive).sum() / positive_weights.sum())


def test_weighted_cluster_bootstrap_ap_matches_expanded_rows() -> None:
    labels = np.asarray([1, 0, 0, 1, 1, 0], dtype=np.float64)
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    clusters = np.asarray([0, 0, 1, 1, 2, 2])
    counts = np.asarray([[1, 1, 1], [2, 0, 1], [0, 2, 1]])
    actual = weighted_cluster_bootstrap_ap(labels, scores, clusters, counts)
    expected = np.asarray(
        [_weighted_ap(labels, scores, clusters, row) for row in counts]
    )
    assert np.allclose(actual, expected)


def test_weighted_cluster_bootstrap_ap_groups_exact_ties() -> None:
    labels = np.asarray([1, 0, 0, 1], dtype=np.float64)
    scores = np.zeros(4)
    clusters = np.asarray([0, 0, 1, 1])
    counts = np.asarray([[1, 1], [2, 1], [1, 2]])
    actual = weighted_cluster_bootstrap_ap(labels, scores, clusters, counts)
    assert np.allclose(actual, 0.5)


def test_segment_validity_requires_all_four_target_steps() -> None:
    valid = np.asarray(
        [
            [True] * 8,
            [True] * 7 + [False],
            [True] * 3 + [False] * 5,
        ]
    )
    assert segment_validity(valid).tolist() == [
        [True, True],
        [True, False],
        [False, False],
    ]


def test_arr_detects_true_alignment_and_shuffled_redirection() -> None:
    samples = 4
    classes = len(ACTION_LABEL_NAMES)
    true_labels = np.empty((samples, 2, classes), dtype=np.float32)
    for sample in range(samples):
        for segment in range(2):
            flat = 2 * sample + segment
            true_labels[sample, segment] = [
                (flat + class_index) % 2 for class_index in range(classes)
            ]
    shuffled_labels = 1 - true_labels
    zero_labels = np.zeros_like(true_labels)
    labels = np.stack((true_labels, shuffled_labels, zero_labels))
    real_scores = 0.1 + 0.8 * true_labels
    generated = np.empty((2, 3, samples, 2, classes), dtype=np.float32)
    generated[:, 0] = real_scores
    generated[:, 1] = 0.1 + 0.8 * shuffled_labels
    generated[:, 2] = 0.5

    summary = summarize_arr(
        real_scores=real_scores,
        generated_scores=generated,
        labels_by_mode=labels,
        valid_segments=np.ones((samples, 2), dtype=np.bool_),
        action_modes=["true", "shuffled", "zeros"],
        round_ids=[f"r{index}" for index in range(samples)],
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )
    assert summary["modes"]["true"]["macro_own_conditioning_arr"] == 1.0
    assert summary["modes"]["shuffled"]["macro_own_conditioning_arr"] == 1.0
    assert summary["primary"]["true_target_alignment_separation"]["estimate"] > 0
    assert summary["primary"]["shuffled_action_redirection"]["estimate"] > 0
