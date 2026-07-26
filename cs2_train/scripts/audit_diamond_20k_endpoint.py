"""Independently audit a completed resource-amended DIAMOND endpoint.

This checker does not recompute model metrics. It verifies the immutable
checkpoint/evaluation contract, hashes every rollout-archive payload, and
proves that pixel and motion JSONL files contain the same complete paired grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

WINDOW_MODES = ("midpoint", "first-death")
CONTRACT_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_step",
    "config_sha256",
    "manifest_sha256",
    "map_slug",
    "split",
    "target_fps",
    "resize",
    "rollout_steps",
    "rollout_target_masking",
    "eval_seeds",
    "action_modes",
    "num_eval_samples",
    "num_rounds",
    "weights",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required artifact is absent: {path}") from None
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def load_cells(path: Path) -> tuple[set[tuple[str, int, str]], set[str], int]:
    cells: set[tuple[str, int, str]] = set()
    rounds: set[str] = set()
    row_count = 0
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"required artifact is absent: {path}") from None
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                cell = (
                    str(row["sample_key"]),
                    int(row["eval_seed"]),
                    str(row["action_mode"]),
                )
                round_id = str(row["round_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid row {line_number} in {path}: {error}") from error
            if cell in cells:
                raise ValueError(f"duplicate paired cell in {path}: {cell}")
            cells.add(cell)
            rounds.add(round_id)
            row_count += 1
    return cells, rounds, row_count


def assert_finite_tree(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, label=f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value at {label}: {value}")


def audit_endpoint(
    endpoint_root: Path,
    *,
    expected_step: int = 20_000,
    expected_samples: int = 690,
    expected_rounds: int = 69,
    expected_eval_seeds: tuple[int, ...] = (37, 41, 43),
    expected_action_modes: tuple[str, ...] = ("true", "shuffled", "zeros"),
    checkpoint: Path | None = None,
    probe_checkpoint: Path | None = None,
) -> dict[str, Any]:
    endpoint_root = endpoint_root.resolve()
    if not (endpoint_root / "COMPLETE").is_file():
        raise FileNotFoundError(f"completion marker is absent: {endpoint_root / 'COMPLETE'}")

    expected_cells = expected_samples * len(expected_eval_seeds) * len(expected_action_modes)
    summaries: dict[str, dict[str, Any]] = {}
    windows: dict[str, dict[str, Any]] = {}

    for window in WINDOW_MODES:
        evaluation = endpoint_root / window
        summary_path = evaluation / "summary.json"
        motion_path = evaluation / "motion_metrics" / "summary.json"
        archive_metadata_path = evaluation / "rollout_archive" / "metadata.json"
        metric_rows_path = evaluation / "per_sample_metrics.jsonl"
        motion_rows_path = evaluation / "motion_metrics" / "per_sample_motion_metrics.jsonl"

        summary = load_json(summary_path)
        motion = load_json(motion_path)
        archive = load_json(archive_metadata_path)
        summaries[window] = summary

        expected_contract = {
            "checkpoint_step": expected_step,
            "num_eval_samples": expected_samples,
            "num_rounds": expected_rounds,
            "eval_seeds": list(expected_eval_seeds),
            "action_modes": list(expected_action_modes),
            "window_mode": window,
        }
        for field, expected in expected_contract.items():
            actual = summary.get(field)
            if actual != expected:
                raise ValueError(f"{window} summary {field}={actual!r}, expected {expected!r}")

        if motion.get("status") != "complete":
            raise ValueError(f"{window} motion status is not complete")
        motion_contract = {
            "num_samples": expected_samples,
            "num_rounds": expected_rounds,
            "eval_seeds": list(expected_eval_seeds),
            "action_modes": list(expected_action_modes),
            "sample_plan_sha256": summary["sample_plan_sha256"],
        }
        for field, expected in motion_contract.items():
            actual = motion.get(field)
            if actual != expected:
                raise ValueError(f"{window} motion {field}={actual!r}, expected {expected!r}")

        if archive.get("status") != "complete":
            raise ValueError(f"{window} rollout archive status is not complete")
        archive_contract = archive.get("contract")
        if not isinstance(archive_contract, dict):
            raise TypeError(f"{window} rollout archive has no contract object")
        for field, expected in (
            ("checkpoint_sha256", summary["checkpoint_sha256"]),
            ("checkpoint_step", expected_step),
            ("config_sha256", summary["config_sha256"]),
            ("manifest_sha256", summary["manifest_sha256"]),
            ("sample_plan_sha256", summary["sample_plan_sha256"]),
            ("num_samples", expected_samples),
            ("eval_seeds", list(expected_eval_seeds)),
            ("action_modes", list(expected_action_modes)),
            ("window_mode", window),
        ):
            actual = archive_contract.get(field)
            if actual != expected:
                raise ValueError(
                    f"{window} rollout archive contract {field}={actual!r}, expected {expected!r}"
                )

        archive_metadata_sha256 = sha256_file(archive_metadata_path)
        if motion.get("archive_metadata_sha256") != archive_metadata_sha256:
            raise ValueError(f"{window} motion summary references a different rollout archive")

        artifacts = archive.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f"{window} rollout archive has no artifact records")
        artifact_audit = {}
        for name, record in artifacts.items():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise TypeError(f"{window} invalid rollout artifact record: {name}")
            artifact_path = archive_metadata_path.parent / record["path"]
            if artifact_path.parent.resolve() != archive_metadata_path.parent.resolve():
                raise ValueError(
                    f"{window} rollout artifact escapes archive directory: {artifact_path}"
                )
            if not artifact_path.is_file():
                raise FileNotFoundError(f"rollout artifact is absent: {artifact_path}")
            digest = sha256_file(artifact_path)
            if digest != record.get("sha256"):
                raise ValueError(f"{window} rollout artifact hash mismatch: {artifact_path}")
            artifact_audit[name] = {
                "path": record["path"],
                "size_bytes": artifact_path.stat().st_size,
                "sha256": digest,
                "dtype": record.get("dtype"),
                "shape": record.get("shape"),
            }

        temporary_files = sorted(
            str(path.relative_to(evaluation))
            for path in evaluation.rglob("*")
            if path.is_file() and (path.name.startswith(".") or ".tmp" in path.name)
        )
        if temporary_files:
            raise ValueError(f"{window} contains incomplete temporary files: {temporary_files}")

        metric_cells, metric_rounds, metric_row_count = load_cells(metric_rows_path)
        motion_cells, motion_rounds, motion_row_count = load_cells(motion_rows_path)
        if metric_row_count != expected_cells:
            raise ValueError(
                f"{window} has {metric_row_count} metric rows, expected {expected_cells}"
            )
        if motion_row_count != expected_cells:
            raise ValueError(
                f"{window} has {motion_row_count} motion rows, expected {expected_cells}"
            )
        if metric_cells != motion_cells:
            raise ValueError(f"{window} metric and motion paired-cell grids differ")
        if len(metric_rounds) != expected_rounds or motion_rounds != metric_rounds:
            raise ValueError(
                f"{window} metric/motion round clusters are incomplete or inconsistent"
            )

        assert_finite_tree(summary.get("means"), label=f"{window}.summary.means")
        assert_finite_tree(summary.get("paired_deltas"), label=f"{window}.summary.paired_deltas")
        assert_finite_tree(motion.get("means"), label=f"{window}.motion.means")
        assert_finite_tree(motion.get("paired_deltas"), label=f"{window}.motion.paired_deltas")

        window_audit = {
            "summary_sha256": sha256_file(summary_path),
            "motion_summary_sha256": sha256_file(motion_path),
            "archive_metadata_sha256": archive_metadata_sha256,
            "metric_rows_sha256": sha256_file(metric_rows_path),
            "motion_rows_sha256": sha256_file(motion_rows_path),
            "paired_cells": metric_row_count,
            "round_clusters": len(metric_rounds),
            "artifacts": artifact_audit,
        }
        if probe_checkpoint is not None:
            arr_path = evaluation / "action_recoverability" / "summary.json"
            arr = load_json(arr_path)
            if arr.get("status") != "complete":
                raise ValueError(f"{window} action-recoverability status is not complete")
            actual_probe_sha256 = sha256_file(probe_checkpoint)
            arr_contract = {
                "archive_metadata_sha256": archive_metadata_sha256,
                "sample_plan_sha256": summary["sample_plan_sha256"],
                "probe_checkpoint_sha256": actual_probe_sha256,
                "num_samples": expected_samples,
                "eval_seeds": list(expected_eval_seeds),
                "action_modes": list(expected_action_modes),
            }
            for field, expected in arr_contract.items():
                actual = arr.get(field)
                if actual != expected:
                    raise ValueError(
                        f"{window} action recoverability {field}={actual!r}, expected {expected!r}"
                    )
            bootstrap = arr.get("results", {}).get("bootstrap", {})
            if (
                bootstrap.get("unit") != "round_id"
                or bootstrap.get("round_clusters") != expected_rounds
                or bootstrap.get("replicates") != 10_000
            ):
                raise ValueError(
                    f"{window} action-recoverability bootstrap contract is invalid: {bootstrap}"
                )
            arr_artifacts = arr.get("artifacts")
            if not isinstance(arr_artifacts, dict) or not arr_artifacts:
                raise ValueError(f"{window} action-recoverability artifacts are absent")
            arr_artifact_audit = {}
            for name, record in arr_artifacts.items():
                if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                    raise TypeError(f"{window} invalid action-recoverability artifact: {name}")
                artifact_path = arr_path.parent / record["path"]
                if artifact_path.parent.resolve() != arr_path.parent.resolve():
                    raise ValueError(
                        f"{window} action-recoverability artifact escapes result directory: "
                        f"{artifact_path}"
                    )
                digest = sha256_file(artifact_path)
                if digest != record.get("sha256"):
                    raise ValueError(
                        f"{window} action-recoverability artifact hash mismatch: {artifact_path}"
                    )
                arr_artifact_audit[name] = {
                    "path": record["path"],
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": digest,
                    "dtype": record.get("dtype"),
                    "shape": record.get("shape"),
                }
            assert_finite_tree(
                arr.get("results", {}).get("primary"),
                label=f"{window}.action_recoverability.primary",
            )
            window_audit["action_recoverability"] = {
                "summary_sha256": sha256_file(arr_path),
                "probe_checkpoint_sha256": actual_probe_sha256,
                "complete_segments": arr.get("num_complete_segments"),
                "incomplete_segments_excluded": arr.get("num_incomplete_segments_excluded"),
                "artifacts": arr_artifact_audit,
            }
        windows[window] = window_audit

    first = summaries[WINDOW_MODES[0]]
    for window in WINDOW_MODES[1:]:
        for field in CONTRACT_FIELDS:
            if summaries[window].get(field) != first.get(field):
                raise ValueError(f"cross-window contract mismatch for {field}")

    checkpoint_sha256 = first["checkpoint_sha256"]
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        actual_checkpoint_sha256 = sha256_file(checkpoint)
        if actual_checkpoint_sha256 != checkpoint_sha256:
            raise ValueError(
                f"checkpoint hash {actual_checkpoint_sha256} != summary hash {checkpoint_sha256}"
            )

    return {
        "schema_version": 1,
        "status": "pass",
        "purpose": "post_test_artifact_integrity_audit",
        "endpoint_root": str(endpoint_root),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_rehashed": checkpoint is not None,
        "expected_step": expected_step,
        "expected_samples": expected_samples,
        "expected_rounds": expected_rounds,
        "expected_eval_seeds": list(expected_eval_seeds),
        "expected_action_modes": list(expected_action_modes),
        "cross_window_contract_fields": list(CONTRACT_FIELDS),
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe-checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, default=20_000)
    parser.add_argument("--expected-samples", type=int, default=690)
    parser.add_argument("--expected-rounds", type=int, default=69)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[37, 41, 43])
    parser.add_argument(
        "--action-modes",
        nargs="+",
        default=["true", "shuffled", "zeros"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = audit_endpoint(
        args.endpoint_root,
        expected_step=args.expected_step,
        expected_samples=args.expected_samples,
        expected_rounds=args.expected_rounds,
        expected_eval_seeds=tuple(args.eval_seeds),
        expected_action_modes=tuple(args.action_modes),
        checkpoint=args.checkpoint,
        probe_checkpoint=args.probe_checkpoint,
    )
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
