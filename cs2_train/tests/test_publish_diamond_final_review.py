from __future__ import annotations

import json

from cs2_train.scripts.publish_diamond_final_review import (
    LOCAL_PATH_FIELDS,
    write_sanitized_summary,
)


def test_write_sanitized_summary_drops_host_paths(tmp_path) -> None:
    source = {
        "checkpoint": "/host/run/latest.pt",
        "config": "/host/repo/config.json",
        "manifest": "/host/data/manifest.parquet",
        "sample_plan": "/host/run/sample-plan.json",
        "review_manifest": "/host/run/review-manifest.json",
        "per_sample_metrics": "/host/run/rows.jsonl",
        "checkpoint_sha256": "abc",
        "means": {"true": {"rollout_mse_mean": 0.1}},
    }
    output = write_sanitized_summary(source, tmp_path / "summary.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert not LOCAL_PATH_FIELDS.intersection(payload)
    assert payload["checkpoint_sha256"] == "abc"
    assert payload["means"] == source["means"]
