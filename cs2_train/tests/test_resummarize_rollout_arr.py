from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cs2_train.scripts.resummarize_rollout_arr import ARRAY_NAMES, build_addendum
from cs2_train.src.evaluate_rollout_arr import summarize_arr
from cs2_train.src.rollout_archive import sha256_file
from cs2_train.src.temporal_action_probe import ACTION_LABEL_NAMES


def test_arr_addendum_verifies_saved_arrays_and_adds_per_action_intervals(
    tmp_path: Path,
) -> None:
    samples = 4
    classes = len(ACTION_LABEL_NAMES)
    true_labels = np.empty((samples, 2, classes), dtype=np.float32)
    for sample in range(samples):
        for segment in range(2):
            flat = 2 * sample + segment
            true_labels[sample, segment] = [
                (flat + class_index) % 2 for class_index in range(classes)
            ]
    labels = np.stack((true_labels, 1 - true_labels, np.zeros_like(true_labels)))
    real_scores = 0.1 + 0.8 * true_labels
    generated_scores = np.empty((2, 3, samples, 2, classes), dtype=np.float32)
    generated_scores[:, 0] = real_scores
    generated_scores[:, 1] = 0.1 + 0.8 * (1 - true_labels)
    generated_scores[:, 2] = 0.5
    valid_segments = np.ones((samples, 2), dtype=np.bool_)
    round_ids = [f"round-{index}" for index in range(samples)]
    results = summarize_arr(
        real_scores=real_scores,
        generated_scores=generated_scores,
        labels_by_mode=labels,
        valid_segments=valid_segments,
        action_modes=["true", "shuffled", "zeros"],
        round_ids=round_ids,
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )
    results.pop("per_class_primary")

    arr_dir = tmp_path / "arr"
    arr_dir.mkdir()
    arrays = dict(
        zip(
            ARRAY_NAMES,
            (real_scores, generated_scores, labels, valid_segments),
            strict=True,
        )
    )
    artifacts = {}
    for name, array in arrays.items():
        path = arr_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        artifacts[name] = {"path": path.name, "sha256": sha256_file(path)}
    sample_plan = tmp_path / "sample_plan.json"
    sample_plan.write_text(
        json.dumps([{"round_id": round_id} for round_id in round_ids]),
        encoding="utf-8",
    )
    (arr_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "num_samples": samples,
                "sample_plan_sha256": sha256_file(sample_plan),
                "action_modes": ["true", "shuffled", "zeros"],
                "label_names": list(ACTION_LABEL_NAMES),
                "artifacts": artifacts,
                "results": results,
            }
        ),
        encoding="utf-8",
    )

    addendum = build_addendum(arr_dir, sample_plan)
    assert addendum["source_results_recomputed_exactly"] is True
    assert set(addendum["per_class_primary"]) == set(ACTION_LABEL_NAMES)
    fire = addendum["per_class_primary"]["FIRE"]
    assert fire["true_minus_shuffled_target_alignment"]["ci95"][0] > 0
