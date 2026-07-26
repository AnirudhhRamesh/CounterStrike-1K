#!/usr/bin/env python3
"""Add per-action round-bootstrap intervals to an existing verified ARR result.

This does not rerun the visual backbone or probe. It verifies and reuses the exact saved score,
label, validity, and sample-plan arrays produced by ``evaluate_rollout_arr``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from cs2_train.src.evaluate_rollout_arr import summarize_arr
from cs2_train.src.rollout_archive import sha256_file

ARRAY_NAMES = (
    "real_scores_float32",
    "generated_scores_float32",
    "labels_by_mode_float32",
    "valid_segments_bool",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def build_addendum(arr_dir: Path, sample_plan_path: Path) -> dict[str, Any]:
    source_path = arr_dir / "summary.json"
    source = read_json(source_path)
    if source.get("status") != "complete":
        raise ValueError("ARR addendum requires a complete full-result source summary")
    if sha256_file(sample_plan_path) != source.get("sample_plan_sha256"):
        raise ValueError("sample-plan hash does not match the ARR source summary")

    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        metadata = source.get("artifacts", {}).get(name)
        if not isinstance(metadata, dict):
            raise TypeError(f"ARR source summary is missing artifact metadata for {name}")
        path = arr_dir / str(metadata["path"])
        if sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"ARR artifact hash mismatch: {name}")
        arrays[name] = np.load(path, allow_pickle=False)

    sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))
    if not isinstance(sample_plan, list) or len(sample_plan) != source.get("num_samples"):
        raise ValueError("sample plan does not match the ARR source row count")
    round_ids = [str(row["round_id"]) for row in sample_plan]
    bootstrap = source["results"]["bootstrap"]
    recomputed = summarize_arr(
        real_scores=arrays["real_scores_float32"],
        generated_scores=arrays["generated_scores_float32"],
        labels_by_mode=arrays["labels_by_mode_float32"],
        valid_segments=arrays["valid_segments_bool"],
        action_modes=list(source["action_modes"]),
        round_ids=round_ids,
        bootstrap_seed=int(bootstrap["seed"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
    )

    for key, value in source["results"].items():
        if key == "per_class_primary":
            continue
        if recomputed.get(key) != value:
            raise ValueError(f"recomputed ARR result drifted at {key}")

    return {
        "schema": "cs2-rollout-arr-per-action-addendum-v1",
        "status": "complete",
        "purpose": "all-action paired round-bootstrap uncertainty without probe rescoring",
        "source_summary": str(source_path.resolve()),
        "source_summary_sha256": sha256_file(source_path),
        "sample_plan": str(sample_plan_path.resolve()),
        "sample_plan_sha256": sha256_file(sample_plan_path),
        "source_results_recomputed_exactly": True,
        "action_modes": list(source["action_modes"]),
        "label_names": list(source["label_names"]),
        "per_class_primary": recomputed["per_class_primary"],
        "bootstrap": recomputed["bootstrap"],
        "source_artifacts": {
            name: {
                "path": str((arr_dir / source["artifacts"][name]["path"]).resolve()),
                "sha256": source["artifacts"][name]["sha256"],
            }
            for name in ARRAY_NAMES
        },
    }


def git_output(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arr_dir", type=Path)
    parser.add_argument("--sample-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    project_root = Path(__file__).resolve().parents[2]
    code_status = git_output(project_root, "status", "--porcelain=v1")
    if code_status:
        raise RuntimeError("publication ARR resummarization requires a clean source tree")
    payload = build_addendum(args.arr_dir.resolve(), args.sample_plan.resolve())
    payload["code_commit"] = git_output(project_root, "rev-parse", "HEAD")
    payload["code_status"] = code_status

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
