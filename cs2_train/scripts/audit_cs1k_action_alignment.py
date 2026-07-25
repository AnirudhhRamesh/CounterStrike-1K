"""Audit CounterStrike-1K target-frame action alignment.

The release stores mouse deltas on the frame they produce:

    action[i].mouse == state[i].view_angle - state[i - 1].view_angle

At 8 fps, an action conditioning observation at source frame ``t`` must
therefore aggregate source action rows ``t + 1`` through ``t + 4`` to predict
the next emitted observation at source frame ``t + 4``. This script verifies
that invariant on a deterministic Dust2 panel before a rebuttal run starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from counterstrike1k.schema import ACTIONS_DTYPE, STATE_DTYPE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrapped_degrees(value: np.ndarray) -> np.ndarray:
    return (value + 180.0) % 360.0 - 180.0


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def audit_sample(
    data_dir: Path,
    sample_key: str,
    *,
    source_stride: int,
    tolerance: float,
    valid_end_frame: int | None = None,
) -> dict:
    actions_path = data_dir / "actions" / f"{sample_key}.actions.bin"
    state_path = data_dir / "state" / f"{sample_key}.state.bin"
    actions = np.fromfile(actions_path, dtype=ACTIONS_DTYPE)
    state = np.fromfile(state_path, dtype=STATE_DTYPE)
    if len(actions) != len(state):
        raise ValueError(
            f"{sample_key}: action/state length mismatch {len(actions)} != {len(state)}"
        )
    if len(actions) <= source_stride:
        raise ValueError(
            f"{sample_key}: only {len(actions)} frames; need > {source_stride}"
        )
    if not np.array_equal(actions["tick"], state["tick"]):
        raise ValueError(f"{sample_key}: action and state ticks differ")

    valid_end = (
        len(actions)
        if valid_end_frame is None
        else min(len(actions), int(valid_end_frame))
    )
    if valid_end <= source_stride:
        raise ValueError(
            f"{sample_key}: valid interval has only {valid_end} frames; "
            f"need > {source_stride}"
        )
    pitch_step_error = (
        actions["delta_pitch"][1:valid_end].astype(np.float64)
        - np.diff(state["pitch"][:valid_end].astype(np.float64))
    )
    yaw_step_error = _wrapped_degrees(
        actions["delta_yaw"][1:valid_end].astype(np.float64)
        - _wrapped_degrees(np.diff(state["yaw"][:valid_end].astype(np.float64)))
    )

    # Verify the exact transition consumed by the 32->8 fps DIAMOND adapter:
    # observation t -- actions[t+1:t+stride+1] --> observation t+stride.
    starts = np.arange(0, valid_end - source_stride, source_stride)
    pitch_interval = np.asarray(
        [
            actions["delta_pitch"][start + 1 : start + source_stride + 1].sum(
                dtype=np.float64
            )
            for start in starts
        ]
    )
    yaw_interval = np.asarray(
        [
            actions["delta_yaw"][start + 1 : start + source_stride + 1].sum(
                dtype=np.float64
            )
            for start in starts
        ]
    )
    pitch_transition = (
        state["pitch"][starts + source_stride].astype(np.float64)
        - state["pitch"][starts].astype(np.float64)
    )
    yaw_transition = _wrapped_degrees(
        state["yaw"][starts + source_stride].astype(np.float64)
        - state["yaw"][starts].astype(np.float64)
    )
    pitch_interval_error = pitch_interval - pitch_transition
    yaw_interval_error = _wrapped_degrees(yaw_interval - yaw_transition)

    post_valid_mismatches = 0
    if valid_end < len(actions):
        post_pitch_error = (
            actions["delta_pitch"][valid_end:].astype(np.float64)
            - np.diff(state["pitch"][valid_end - 1 :].astype(np.float64))
        )
        post_yaw_error = _wrapped_degrees(
            actions["delta_yaw"][valid_end:].astype(np.float64)
            - _wrapped_degrees(
                np.diff(state["yaw"][valid_end - 1 :].astype(np.float64))
            )
        )
        post_valid_mismatches = int(
            np.count_nonzero(
                (np.abs(post_pitch_error) > tolerance)
                | (np.abs(post_yaw_error) > tolerance)
            )
        )

    maxima = {
        "step_pitch_max_abs_error": _max_abs(pitch_step_error),
        "step_yaw_max_abs_error": _max_abs(yaw_step_error),
        "interval_pitch_max_abs_error": _max_abs(pitch_interval_error),
        "interval_yaw_max_abs_error": _max_abs(yaw_interval_error),
    }
    return {
        "sample_key": sample_key,
        "frames": len(actions),
        "valid_end_frame_exclusive": valid_end,
        "post_valid_frames": len(actions) - valid_end,
        "post_valid_step_mismatches": post_valid_mismatches,
        "transitions_checked": len(starts),
        **maxima,
        "passed": all(value <= tolerance for value in maxima.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-slug", default="dust2")
    parser.add_argument("--source-fps", type=int, default=32)
    parser.add_argument("--target-fps", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    if args.source_fps % args.target_fps:
        raise ValueError("target-fps must divide source-fps")
    source_stride = args.source_fps // args.target_fps
    manifest_path = args.data_dir / args.manifest_name
    manifest = pd.read_parquet(manifest_path)
    manifest = manifest[
        manifest["map_slug"].astype(str) == str(args.map_slug)
    ].copy()
    manifest = manifest[
        manifest["split"].astype(str).isin([str(split) for split in args.splits])
    ].copy()
    if manifest.empty:
        raise ValueError(f"{manifest_path}: no map_slug={args.map_slug!r} rows")
    panel = (
        manifest.sort_values(
            ["split", "match_id", "round_idx", "pov_idx"], kind="stable"
        )
        .groupby("split", sort=True, group_keys=False)
        .head(args.max_samples_per_split)
    )
    results = [
        {
            "split": str(row["split"]),
            **audit_sample(
                args.data_dir,
                str(row["sample_key"]),
                source_stride=source_stride,
                tolerance=args.tolerance,
                valid_end_frame=(
                    int(row["alive_end_frame"])
                    if "alive_end_frame" in row
                    and not pd.isna(row["alive_end_frame"])
                    else None
                ),
            ),
        }
        for _, row in panel.iterrows()
    ]
    passed = all(result["passed"] for result in results)
    payload = {
        "passed": passed,
        "manifest": args.manifest_name,
        "manifest_sha256": _sha256(manifest_path),
        "map_slug": args.map_slug,
        "splits": args.splits,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "source_stride": source_stride,
        "tolerance": args.tolerance,
        "samples_checked": len(results),
        "alignment_contract": {
            "observation_source_frame": "t",
            "transition_action_source_frames_inclusive": [
                "t + 1",
                f"t + {source_stride}",
            ],
            "next_observation_source_frame": f"t + {source_stride}",
            "release_mouse_delta_semantics": (
                "action[i] equals wrapped state[i] minus state[i-1]"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("CounterStrike-1K action alignment audit failed")


if __name__ == "__main__":
    main()
