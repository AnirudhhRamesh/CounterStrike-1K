"""Build synchronized video-window manifests for CounterStrike-1K evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import (
    dataframe_sha256,
    filter_manifest_for_subset,
    git_commit,
    load_release_tables,
    write_json,
)


def phase_bucket(round_row: pd.Series, start_tick: int, end_tick: int) -> str:
    round_start = int(round_row["round_start_tick"])
    round_stop = int(round_row["round_stop_tick"])
    if round_stop <= round_start:
        return "unknown"
    mid = 0.5 * (int(start_tick) + int(end_tick))
    frac = (mid - round_start) / float(round_stop - round_start)
    if frac < 0.33:
        return "early"
    if frac < 0.66:
        return "mid"
    return "late"


def _select_rounds(
    rounds: pd.DataFrame,
    *,
    split: str | None,
    map_slug: str | None,
    max_rounds_per_split: int | None,
    seed: int,
) -> pd.DataFrame:
    out = rounds[rounds["complete_10_pov"] == True].copy()  # noqa: E712
    if split:
        out = out[out["split"] == split]
    if map_slug:
        out = out[out["map_slug"] == map_slug]
    out = out.sort_values(["split", "map_slug", "match_id", "round_idx"]).reset_index(drop=True)
    if max_rounds_per_split is None:
        return out
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in out.groupby("split", sort=True, dropna=False):
        if len(group) <= max_rounds_per_split:
            selected.append(group)
            continue
        idx = np.sort(rng.choice(group.index.to_numpy(), size=max_rounds_per_split, replace=False))
        selected.append(group.loc[idx])
    if not selected:
        return out.iloc[0:0]
    return pd.concat(selected).sort_values(["split", "map_slug", "match_id", "round_idx"]).reset_index(drop=True)


def build_windows(
    *,
    root: Path,
    subset: str | None,
    split: str | None,
    map_slug: str | None,
    window_seconds: float,
    windows_per_round: int,
    alive_only: bool,
    max_rounds_per_split: int | None,
    seed: int,
) -> pd.DataFrame:
    manifest, round_index = load_release_tables(root)
    manifest = filter_manifest_for_subset(manifest, root=root, subset=subset)
    round_ids = set(manifest["round_id"].astype(str).tolist())
    rounds = round_index[round_index["round_id"].astype(str).isin(round_ids)].copy()
    rounds = _select_rounds(
        rounds,
        split=split,
        map_slug=map_slug,
        max_rounds_per_split=max_rounds_per_split,
        seed=seed,
    )

    sample_groups = {
        str(round_id): group.sort_values("pov_idx").copy()
        for round_id, group in manifest.groupby("round_id", sort=False)
    }
    rows: list[dict] = []
    for _, round_row in rounds.iterrows():
        round_id = str(round_row["round_id"])
        samples = sample_groups.get(round_id)
        if samples is None or len(samples) != 10:
            continue
        fps = float(samples["fps"].iloc[0])
        frame_stride = int(samples["frame_tick_stride"].iloc[0])
        window_frames = max(2, int(round(window_seconds * fps)))
        window_ticks = window_frames * frame_stride

        start_col = "alive_intersection_start_tick" if alive_only else "clip_intersection_start_tick"
        end_col = "alive_intersection_end_tick" if alive_only else "clip_intersection_end_tick"
        interval_start = int(round_row[start_col])
        interval_end = int(round_row[end_col])
        max_start = interval_end - window_ticks
        if max_start < interval_start:
            continue

        if windows_per_round <= 1:
            starts = [interval_start + (max_start - interval_start) // 2]
        else:
            starts = [
                int(round(v))
                for v in np.linspace(interval_start, max_start, num=windows_per_round)
            ]

        for window_idx, start_tick in enumerate(starts):
            end_tick = int(start_tick + window_ticks)
            eval_window_id = f"{round_id}__w{window_idx:03d}"
            phase = phase_bucket(round_row, start_tick, end_tick)
            valid_samples = []
            for _, sample in samples.iterrows():
                frame0_tick = int(sample["frame0_tick"])
                stride = int(sample["frame_tick_stride"])
                frames = int(sample["frames"])
                start_frame = int(np.floor((int(start_tick) - frame0_tick) / stride))
                end_frame = start_frame + window_frames
                if start_frame < 0 or end_frame > frames:
                    valid_samples = []
                    break
                valid_samples.append((sample, start_frame, end_frame))
            if len(valid_samples) != 10:
                continue
            for sample, start_frame, end_frame in valid_samples:
                rows.append({
                    "eval_window_id": eval_window_id,
                    "round_id": round_id,
                    "match_id": str(round_row["match_id"]),
                    "round_idx": int(round_row["round_idx"]),
                    "window_idx": int(window_idx),
                    "sample_key": str(sample["sample_key"]),
                    "pov_idx": int(sample["pov_idx"]),
                    "split": str(round_row["split"]),
                    "map_slug": str(round_row["map_slug"]),
                    "team_side": str(sample.get("team_side", "")),
                    "phase_bucket": phase,
                    "start_tick": int(start_tick),
                    "end_tick": int(end_tick),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "window_frames": int(window_frames),
                    "fps": float(fps),
                    "frame_tick_stride": int(frame_stride),
                    "alive_only": bool(alive_only),
                    "is_pistol_round": bool(round_row.get("is_pistol_round", False)),
                    "bomb_planted_round": bool(round_row.get("bomb_planted", False)),
                    "bomb_defused_round": bool(round_row.get("bomb_defused", False)),
                    "bomb_exploded_round": bool(round_row.get("bomb_exploded", False)),
                    "round_kills_total": int(round_row.get("round_kills_total", 0)),
                    "has_awp_any_pov": bool(round_row.get("has_awp_any_pov", False)),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--map-slug", default=None)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--windows-per-round", type=int, default=1)
    parser.add_argument("--include-dead-tail", action="store_true",
                        help="Use clip intersection instead of all-players-alive intersection.")
    parser.add_argument("--max-rounds-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output parquet path or directory. Directories receive eval_windows.parquet.")
    args = parser.parse_args()

    out_path = args.out / "eval_windows.parquet" if args.out.suffix == "" else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    windows = build_windows(
        root=args.root,
        subset=args.subset,
        split=args.split,
        map_slug=args.map_slug,
        window_seconds=args.window_seconds,
        windows_per_round=args.windows_per_round,
        alive_only=not args.include_dead_tail,
        max_rounds_per_split=args.max_rounds_per_split,
        seed=args.seed,
    )
    if windows.empty:
        raise RuntimeError("no valid synchronized windows were produced")
    windows.to_parquet(out_path, index=False)
    write_json(out_path.with_suffix(".metadata.json"), {
        "rows": int(len(windows)),
        "rounds": int(windows["round_id"].nunique()),
        "eval_windows": int(windows["eval_window_id"].nunique()),
        "splits": sorted(windows["split"].unique().tolist()),
        "maps": sorted(windows["map_slug"].unique().tolist()),
        "window_seconds": args.window_seconds,
        "windows_per_round": args.windows_per_round,
        "alive_only": not args.include_dead_tail,
        "subset": args.subset,
        "split": args.split,
        "map_slug": args.map_slug,
        "max_rounds_per_split": args.max_rounds_per_split,
        "seed": args.seed,
        "windows_sha256": dataframe_sha256(windows),
        "git_commit": git_commit(),
    })
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
