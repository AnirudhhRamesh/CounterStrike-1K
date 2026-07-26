"""Build causal future-event labels after synchronized video context windows.

Each input video window is treated as context over ``[start_frame, end_frame)``.
Targets are read only from the subsequent interval
``[end_frame, end_frame + horizon_frames)``.  The output has one row per
synchronized ten-POV window, so all event labels are global round events rather
than duplicated per POV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from counterstrike1k.schema import ACTIONS_DTYPE, BUTTONS, STATE_DTYPE

from cs2_release.core.io import (
    DatasetRoots,
    dataframe_sha256,
    git_commit,
    read_member_bytes,
    read_parquet,
    write_json,
)


EVENT_TARGETS = {
    "player_death": "target_PLAYER_DEATH",
    "item_equip": "target_ITEM_EQUIP",
    "weapon_zoom": "target_WEAPON_ZOOM",
    "player_blind": "target_PLAYER_BLIND",
    "bomb_planted": "target_BOMB_PLANTED",
    "bomb_defused": "target_BOMB_DEFUSED",
    "bomb_exploded": "target_BOMB_EXPLODED",
}
DERIVED_TARGETS = (
    "target_FIRE",
    "target_RELOAD",
    "target_DAMAGE",
    "target_WEAPON_SWITCH",
)
TARGET_COLUMNS = (*DERIVED_TARGETS, *EVENT_TARGETS.values())


def _sample_index(roots: DatasetRoots) -> pd.DataFrame | None:
    for candidate in (
        roots.root / f"sample_index_{roots.resolution}.parquet",
        roots.root / "sample_index.parquet",
    ):
        if candidate.exists():
            return read_parquet(candidate)
    return None


def _decode_records(
    sample_key: str,
    suffix: str,
    dtype: np.dtype,
    *,
    roots: DatasetRoots,
    sample_index: pd.DataFrame | None,
) -> np.ndarray:
    payload = read_member_bytes(
        sample_key,
        suffix,
        roots=roots,
        sample_index=sample_index,
    )
    if len(payload) % dtype.itemsize:
        raise ValueError(
            f"{sample_key}.{suffix} has {len(payload)} bytes, "
            f"not a multiple of {dtype.itemsize}"
        )
    return np.frombuffer(payload, dtype=dtype)


def _load_events(
    sample_key: str,
    *,
    roots: DatasetRoots,
    sample_index: pd.DataFrame | None,
) -> list[dict]:
    payload = json.loads(
        read_member_bytes(
            sample_key,
            "events.json",
            roots=roots,
            sample_index=sample_index,
        )
    )
    events = payload.get("events", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise TypeError(f"{sample_key}.events.json does not contain an event list")
    return [event for event in events if isinstance(event, dict)]


def _has_button(actions: np.ndarray, name: str) -> bool:
    bit = BUTTONS.index(name)
    return bool(np.any((actions["buttons"].astype(np.uint16) >> bit) & 1))


def _has_damage(state: np.ndarray, start: int, end: int) -> bool:
    lo = max(0, start - 1)
    health = state["health"][lo:end].astype(np.int16)
    return bool(len(health) >= 2 and np.any(np.diff(health) < 0))


def _has_weapon_switch(state: np.ndarray, start: int, end: int) -> bool:
    lo = max(0, start - 1)
    weapon = state["active_weapon_id"][lo:end].astype(np.int16)
    return bool(len(weapon) >= 2 and np.any(np.diff(weapon) != 0))


def build_future_event_labels(
    windows: pd.DataFrame,
    *,
    roots: DatasetRoots,
    horizon_seconds: float,
) -> pd.DataFrame:
    """Return one leak-free global future-event row per complete context window."""

    required = {
        "eval_window_id",
        "round_id",
        "match_id",
        "sample_key",
        "pov_idx",
        "split",
        "map_slug",
        "start_frame",
        "end_frame",
        "fps",
        "frame_tick_stride",
        "end_tick",
    }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"windows are missing required columns: {missing}")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")

    sample_index = _sample_index(roots)
    rows: list[dict] = []
    for eval_window_id, group in windows.groupby("eval_window_id", sort=False):
        group = group.sort_values("pov_idx").reset_index(drop=True)
        if group["pov_idx"].astype(int).tolist() != list(range(10)):
            continue
        if group["round_id"].astype(str).nunique() != 1:
            raise ValueError(f"{eval_window_id}: synchronized POVs span multiple rounds")
        fps_values = group["fps"].astype(float).unique()
        if len(fps_values) != 1:
            raise ValueError(f"{eval_window_id}: POVs disagree on fps")
        future_frames = max(1, int(round(horizon_seconds * float(fps_values[0]))))
        future_start = int(group["end_frame"].iloc[0])
        future_end = future_start + future_frames
        if group["end_frame"].astype(int).nunique() != 1:
            raise ValueError(f"{eval_window_id}: POVs disagree on context end frame")

        target = {name: 0 for name in TARGET_COLUMNS}
        complete = True
        for _, sample in group.iterrows():
            sample_key = str(sample["sample_key"])
            actions = _decode_records(
                sample_key,
                "actions.bin",
                ACTIONS_DTYPE,
                roots=roots,
                sample_index=sample_index,
            )
            state = _decode_records(
                sample_key,
                "state.bin",
                STATE_DTYPE,
                roots=roots,
                sample_index=sample_index,
            )
            if future_end > len(actions) or future_end > len(state):
                complete = False
                break
            future_actions = actions[future_start:future_end]
            target["target_FIRE"] |= int(_has_button(future_actions, "FIRE"))
            target["target_RELOAD"] |= int(_has_button(future_actions, "RELOAD"))
            target["target_DAMAGE"] |= int(_has_damage(state, future_start, future_end))
            target["target_WEAPON_SWITCH"] |= int(
                _has_weapon_switch(state, future_start, future_end)
            )
        if not complete:
            continue

        # Every POV sidecar contains the round-global event stream. Read one copy.
        anchor_key = str(group["sample_key"].iloc[0])
        for event in _load_events(
            anchor_key,
            roots=roots,
            sample_index=sample_index,
        ):
            event_type = str(event.get("type", ""))
            target_name = EVENT_TARGETS.get(event_type)
            if target_name is None or "frame_idx" not in event:
                continue
            frame_idx = int(event["frame_idx"])
            if future_start <= frame_idx < future_end:
                target[target_name] = 1

        first = group.iloc[0]
        stride = int(first["frame_tick_stride"])
        rows.append(
            {
                "eval_window_id": str(eval_window_id),
                "round_id": str(first["round_id"]),
                "match_id": str(first["match_id"]),
                "split": str(first["split"]),
                "map_slug": str(first["map_slug"]),
                "phase_bucket": str(first.get("phase_bucket", "")),
                "context_start_frame": int(first["start_frame"]),
                "context_end_frame": future_start,
                "future_start_frame": future_start,
                "future_end_frame": future_end,
                "context_end_tick": int(first["end_tick"]),
                "future_end_tick": int(first["end_tick"]) + future_frames * stride,
                "horizon_frames": future_frames,
                "horizon_seconds": float(horizon_seconds),
                **target,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = read_parquet(args.windows)
    roots = DatasetRoots.from_args(
        root=args.root,
        shard_root=args.shard_root,
        resolution=args.resolution,
    )
    labels = build_future_event_labels(
        windows,
        roots=roots,
        horizon_seconds=args.horizon_seconds,
    )
    if labels.empty:
        raise RuntimeError("no complete future-event windows were produced")
    out_path = args.out / "future_event_labels.parquet" if args.out.suffix == "" else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_path, index=False)
    target_cols = [column for column in TARGET_COLUMNS if column in labels.columns]
    write_json(
        out_path.with_suffix(".metadata.json"),
        {
            "schema": "cs1k-causal-future-events-v1",
            "rows": int(len(labels)),
            "matches": int(labels["match_id"].nunique()),
            "rounds": int(labels["round_id"].nunique()),
            "splits": sorted(labels["split"].astype(str).unique().tolist()),
            "maps": sorted(labels["map_slug"].astype(str).unique().tolist()),
            "horizon_seconds": float(args.horizon_seconds),
            "interval_contract": "[context_start, context_end) -> [context_end, future_end)",
            "target_cols": target_cols,
            "target_prevalence": {
                column: float(labels[column].mean()) for column in target_cols
            },
            "windows_sha256": dataframe_sha256(windows),
            "labels_sha256": dataframe_sha256(labels),
            "git_commit": git_commit(),
        },
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
