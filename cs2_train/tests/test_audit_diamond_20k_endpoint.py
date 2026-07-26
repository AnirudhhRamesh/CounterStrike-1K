from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cs2_train.scripts.audit_diamond_20k_endpoint import audit_endpoint


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_endpoint(root: Path) -> None:
    (root / "COMPLETE").parent.mkdir(parents=True, exist_ok=True)
    (root / "COMPLETE").touch()
    common = {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_step": 20_000,
        "config_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "map_slug": "dust2",
        "split": "test",
        "target_fps": 8,
        "resize": [36, 64],
        "rollout_steps": 8,
        "rollout_target_masking": "source_frame < alive_end_frame",
        "eval_seeds": [37],
        "action_modes": ["true", "shuffled"],
        "num_eval_samples": 2,
        "num_rounds": 2,
        "weights": "ema",
    }
    for window in ("midpoint", "first-death"):
        evaluation = root / window
        summary = {
            **common,
            "window_mode": window,
            "sample_plan_sha256": f"{window}-plan",
            "means": {"true": {"loss": 0.1}, "shuffled": {"loss": 0.2}},
            "paired_deltas": {"loss": {"shuffled": {"mean_delta": 0.1}}},
        }
        write_json(evaluation / "summary.json", summary)

        archive_dir = evaluation / "rollout_archive"
        archive_dir.mkdir(parents=True)
        payload = archive_dir / "payload.bin"
        payload.write_bytes(f"{window}-payload".encode())
        archive = {
            "status": "complete",
            "contract": {
                "checkpoint_sha256": common["checkpoint_sha256"],
                "checkpoint_step": 20_000,
                "config_sha256": common["config_sha256"],
                "manifest_sha256": common["manifest_sha256"],
                "sample_plan_sha256": summary["sample_plan_sha256"],
                "num_samples": 2,
                "eval_seeds": [37],
                "action_modes": ["true", "shuffled"],
                "window_mode": window,
            },
            "artifacts": {
                "payload": {
                    "path": payload.name,
                    "sha256": sha256(payload),
                    "dtype": "uint8",
                    "shape": [len(payload.read_bytes())],
                }
            },
        }
        metadata = archive_dir / "metadata.json"
        write_json(metadata, archive)

        rows = []
        for sample_index in range(2):
            for action_mode in ("true", "shuffled"):
                rows.append(
                    {
                        "sample_key": f"sample-{sample_index}",
                        "round_id": f"round-{sample_index}",
                        "eval_seed": 37,
                        "action_mode": action_mode,
                    }
                )
        jsonl = "".join(json.dumps(row) + "\n" for row in rows)
        (evaluation / "per_sample_metrics.jsonl").write_text(jsonl, encoding="utf-8")
        motion_dir = evaluation / "motion_metrics"
        motion_dir.mkdir(parents=True)
        (motion_dir / "per_sample_motion_metrics.jsonl").write_text(jsonl, encoding="utf-8")
        write_json(
            motion_dir / "summary.json",
            {
                "status": "complete",
                "num_samples": 2,
                "num_rounds": 2,
                "eval_seeds": [37],
                "action_modes": ["true", "shuffled"],
                "sample_plan_sha256": summary["sample_plan_sha256"],
                "archive_metadata_sha256": sha256(metadata),
                "means": {"true": {"flow": 0.1}, "shuffled": {"flow": 0.2}},
                "paired_deltas": {"flow": {"shuffled": {"mean_delta": 0.1}}},
            },
        )


def test_audit_endpoint_accepts_complete_hashed_grid(tmp_path: Path) -> None:
    make_endpoint(tmp_path)
    audit = audit_endpoint(
        tmp_path,
        expected_samples=2,
        expected_rounds=2,
        expected_eval_seeds=(37,),
        expected_action_modes=("true", "shuffled"),
    )
    assert audit["status"] == "pass"
    assert audit["windows"]["midpoint"]["paired_cells"] == 4
    assert audit["windows"]["first-death"]["round_clusters"] == 2


def test_audit_endpoint_rejects_archive_corruption(tmp_path: Path) -> None:
    make_endpoint(tmp_path)
    (tmp_path / "midpoint" / "rollout_archive" / "payload.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        audit_endpoint(
            tmp_path,
            expected_samples=2,
            expected_rounds=2,
            expected_eval_seeds=(37,),
            expected_action_modes=("true", "shuffled"),
        )


def test_audit_endpoint_rejects_metric_motion_grid_drift(tmp_path: Path) -> None:
    make_endpoint(tmp_path)
    path = tmp_path / "first-death" / "motion_metrics" / "per_sample_motion_metrics.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="motion rows"):
        audit_endpoint(
            tmp_path,
            expected_samples=2,
            expected_rounds=2,
            expected_eval_seeds=(37,),
            expected_action_modes=("true", "shuffled"),
        )
