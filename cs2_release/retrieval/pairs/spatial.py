"""Build co-located cross-POV retrieval sets from public match state labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import (
    DatasetRoots,
    dataframe_sha256,
    git_commit,
    read_member_bytes,
    read_parquet,
    read_release_parquet,
    write_json,
)


POSITION_COLUMNS = ["pos_x", "pos_y", "pos_z"]

_STATE_BIN_DTYPE = np.dtype([
    ("tick", "<u4"),
    ("pitch", "<f4"),
    ("yaw", "<f4"),
    ("pos_x", "<f4"),
    ("pos_y", "<f4"),
    ("pos_z", "<f4"),
    ("active_weapon", "u1"),
    ("active_weapon_id", "u1"),
    ("ammo_clip", "u1"),
    ("ammo_reserve", "u1"),
    ("health", "u1"),
    ("armor_value", "u1"),
    ("balance", "<u2"),
    ("t_score", "u1"),
    ("ct_score", "u1"),
    ("has_helmet", "u1"),
    ("has_defuser", "u1"),
    ("has_bomb", "u1"),
])


def _state_path(match_id: str) -> Path:
    return Path("match_states") / f"match_{match_id}.parquet"


def _sample_rows(group: pd.DataFrame, n: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if n is None or len(group) <= n:
        return group
    idx = np.sort(rng.choice(group.index.to_numpy(), size=n, replace=False))
    return group.loc[idx]


def attach_midpoint_positions(
    windows: pd.DataFrame,
    *,
    root: Path,
    roots: DatasetRoots | None,
    max_tick_gap: int,
) -> pd.DataFrame:
    if roots is not None:
        rows = _attach_midpoint_positions_from_state_bin(
            windows,
            roots=roots,
            max_tick_gap=max_tick_gap,
        )
        if rows:
            return pd.DataFrame(rows)

    rows: list[dict] = []
    for match_id, match_windows in windows.groupby("match_id", sort=False):
        try:
            state = read_release_parquet(root, _state_path(str(match_id)))
        except FileNotFoundError:
            continue
        if not {"tick", "pov_idx", *POSITION_COLUMNS}.issubset(state.columns):
            continue
        state = state[["tick", "pov_idx", *POSITION_COLUMNS]].dropna().copy()
        state["tick"] = state["tick"].astype(np.int64)
        state["pov_idx"] = state["pov_idx"].astype(np.int16)
        by_pov = {
            int(pov): group.sort_values("tick").reset_index(drop=True)
            for pov, group in state.groupby("pov_idx", sort=False)
        }
        for _, row in match_windows.iterrows():
            pov_idx = int(row["pov_idx"])
            pov_state = by_pov.get(pov_idx)
            if pov_state is None or pov_state.empty:
                continue
            ticks = pov_state["tick"].to_numpy(dtype=np.int64)
            mid_tick = int((int(row["start_tick"]) + int(row["end_tick"])) // 2)
            idx = int(np.searchsorted(ticks, mid_tick, side="left"))
            candidates = []
            if idx < len(ticks):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            if not candidates:
                continue
            best_idx = min(candidates, key=lambda i: abs(int(ticks[i]) - mid_tick))
            tick_gap = abs(int(ticks[best_idx]) - mid_tick)
            if tick_gap > max_tick_gap:
                continue
            out = row.to_dict()
            out["mid_tick"] = mid_tick
            out["state_tick"] = int(ticks[best_idx])
            out["state_tick_gap"] = int(tick_gap)
            for col in POSITION_COLUMNS:
                out[col] = float(pov_state.iloc[best_idx][col])
            rows.append(out)
    return pd.DataFrame(rows)


def _sample_index_for(roots: DatasetRoots) -> pd.DataFrame | None:
    for candidate in (
        roots.root / f"sample_index_{roots.resolution}.parquet",
        roots.root / "sample_index.parquet",
    ):
        if candidate.exists():
            return read_parquet(candidate)
    return None


def _attach_midpoint_positions_from_state_bin(
    windows: pd.DataFrame,
    *,
    roots: DatasetRoots,
    max_tick_gap: int,
) -> list[dict]:
    sample_index = _sample_index_for(roots)
    rows: list[dict] = []
    for _, row in windows.iterrows():
        sample_key = str(row["sample_key"])
        try:
            payload = read_member_bytes(
                sample_key,
                "state.bin",
                roots=roots,
                sample_index=sample_index,
            )
        except (FileNotFoundError, ValueError):
            continue
        if len(payload) < _STATE_BIN_DTYPE.itemsize:
            continue
        n_records = len(payload) // _STATE_BIN_DTYPE.itemsize
        state = np.frombuffer(payload, dtype=_STATE_BIN_DTYPE, count=n_records)
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        if start_frame < 0 or end_frame <= start_frame or start_frame >= n_records:
            continue
        mid_frame = min(max((start_frame + end_frame) // 2, 0), n_records - 1)
        record = state[mid_frame]
        mid_tick = int((int(row["start_tick"]) + int(row["end_tick"])) // 2)
        state_tick = int(record["tick"])
        tick_gap = abs(state_tick - mid_tick)
        if tick_gap > max_tick_gap:
            continue
        positions = [float(record[col]) for col in POSITION_COLUMNS]
        if not all(np.isfinite(positions)):
            continue
        out = row.to_dict()
        out["mid_tick"] = mid_tick
        out["state_tick"] = state_tick
        out["state_tick_gap"] = int(tick_gap)
        for col, value in zip(POSITION_COLUMNS, positions, strict=True):
            out[col] = value
        rows.append(out)
    return rows


def _distance(a: pd.Series, b: pd.Series, *, include_z: bool) -> float:
    cols = POSITION_COLUMNS if include_z else POSITION_COLUMNS[:2]
    av = a[cols].to_numpy(dtype=np.float32)
    bv = b[cols].to_numpy(dtype=np.float32)
    return float(np.linalg.norm(av - bv))


def _distances_to_query(query: pd.Series, candidates: pd.DataFrame, *, include_z: bool) -> np.ndarray:
    cols = POSITION_COLUMNS if include_z else POSITION_COLUMNS[:2]
    qv = query[cols].to_numpy(dtype=np.float32)
    cv = candidates[cols].to_numpy(dtype=np.float32)
    return np.linalg.norm(cv - qv[None, :], axis=1).astype(np.float32, copy=False)


def build_spatial_retrieval_pairs(
    windows: pd.DataFrame,
    *,
    root: Path,
    roots: DatasetRoots | None,
    split: str,
    candidates_per_query: int,
    max_queries: int | None,
    max_positives: int | None,
    positive_radius: float,
    negative_policy: str,
    negative_min_radius: float | None,
    negative_location_radius: float | None,
    same_team_only: bool,
    include_z: bool,
    max_tick_gap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = windows[windows["split"] == split].copy().reset_index(drop=True)
    df = attach_midpoint_positions(df, root=root, roots=roots, max_tick_gap=max_tick_gap)
    if df.empty:
        return df
    df = df.reset_index(drop=True)
    df["window_row_id"] = np.arange(len(df), dtype=np.int64)
    groups = {str(k): g.copy() for k, g in df.groupby("eval_window_id", sort=False)}
    query_rows = df.sort_values(["map_slug", "phase_bucket", "round_id", "pov_idx"]).copy()
    query_rows = _sample_rows(query_rows, max_queries, rng)

    pair_rows: list[dict] = []
    for _, query in query_rows.iterrows():
        same_window = groups[str(query["eval_window_id"])]
        positives = same_window[same_window["pov_idx"] != int(query["pov_idx"])].copy()
        if same_team_only and "team_side" in positives.columns:
            positives = positives[positives["team_side"].astype(str) == str(query.get("team_side", ""))]
        if positives.empty:
            continue
        positives["query_candidate_distance"] = _distances_to_query(query, positives, include_z=include_z)
        same_time_distances = positives.copy()
        positives = positives[positives["query_candidate_distance"] <= positive_radius]
        if positives.empty and negative_policy == "same_time_far_location":
            positives = same_time_distances.sort_values(["query_candidate_distance", "pov_idx"]).head(1)
        if positives.empty:
            continue
        positives = positives.sort_values(["query_candidate_distance", "pov_idx"])
        positives = _sample_rows(positives, max_positives, rng)
        if len(positives) >= candidates_per_query:
            positives = positives.head(candidates_per_query - 1)

        if negative_policy == "same_time_far_location":
            min_radius = positive_radius if negative_min_radius is None else float(negative_min_radius)
            neg_pool = same_window[same_window["pov_idx"] != int(query["pov_idx"])].copy()
            if same_team_only and "team_side" in neg_pool.columns:
                neg_pool = neg_pool[neg_pool["team_side"].astype(str) == str(query.get("team_side", ""))]
            if not neg_pool.empty:
                neg_pool["query_candidate_distance"] = _distances_to_query(query, neg_pool, include_z=include_z)
                positive_povs = set(positives["pov_idx"].astype(int).tolist())
                neg_pool = neg_pool[
                    (~neg_pool["pov_idx"].astype(int).isin(positive_povs))
                    & (neg_pool["query_candidate_distance"] > min_radius)
                ]
                if neg_pool.empty:
                    neg_pool = same_time_distances[
                        ~same_time_distances["pov_idx"].astype(int).isin(positive_povs)
                    ].sort_values(["query_candidate_distance", "pov_idx"], ascending=[False, True])
        elif negative_policy == "same_map_phase_different_round":
            neg_pool = df[
                (df["map_slug"] == query["map_slug"])
                & (df["phase_bucket"] == query["phase_bucket"])
                & (df["round_id"] != query["round_id"])
            ].copy()
            if neg_pool.empty:
                neg_pool = df[
                    (df["map_slug"] == query["map_slug"])
                    & (df["round_id"] != query["round_id"])
                ].copy()
        elif negative_policy == "same_location_wrong_time":
            location_radius = (
                float(positive_radius)
                if negative_location_radius is None
                else float(negative_location_radius)
            )
            neg_pool = df[
                (df["map_slug"] == query["map_slug"])
                & (df["phase_bucket"] == query["phase_bucket"])
                & (df["round_id"] != query["round_id"])
            ].copy()
            if not neg_pool.empty:
                neg_pool["query_candidate_distance"] = _distances_to_query(query, neg_pool, include_z=include_z)
                neg_pool = neg_pool[neg_pool["query_candidate_distance"] <= location_radius]
            if neg_pool.empty:
                neg_pool = df[
                    (df["map_slug"] == query["map_slug"])
                    & (df["round_id"] != query["round_id"])
                ].copy()
                if not neg_pool.empty:
                    neg_pool["query_candidate_distance"] = _distances_to_query(query, neg_pool, include_z=include_z)
                    neg_pool = neg_pool[neg_pool["query_candidate_distance"] <= location_radius]
        else:
            raise ValueError(f"unknown negative_policy={negative_policy!r}")
        if neg_pool.empty:
            continue
        n_neg = max(1, candidates_per_query - len(positives))
        replace = len(neg_pool) < n_neg
        neg_indices = rng.choice(neg_pool.index.to_numpy(), size=n_neg, replace=replace)
        candidate_set_id = f"{query['eval_window_id']}__q{int(query['pov_idx']):02d}"
        candidates = [(row, 1, float(row["query_candidate_distance"])) for _, row in positives.iterrows()]
        for idx in neg_indices:
            candidate = neg_pool.loc[idx]
            if "query_candidate_distance" in candidate:
                distance = float(candidate["query_candidate_distance"])
            else:
                distance = _distance(query, candidate, include_z=include_z)
            candidates.append((candidate, 0, distance))
        rng.shuffle(candidates)
        for rank_idx, (candidate, label, distance) in enumerate(candidates):
            pair_rows.append({
                "candidate_set_id": candidate_set_id,
                "query_window_row_id": int(query["window_row_id"]),
                "candidate_window_row_id": int(candidate["window_row_id"]),
                "label": int(label),
                "candidate_ordinal": int(rank_idx),
                "split": split,
                "map_slug": str(query["map_slug"]),
                "phase_bucket": str(query["phase_bucket"]),
                "query_eval_window_id": str(query["eval_window_id"]),
                "candidate_eval_window_id": str(candidate["eval_window_id"]),
                "query_match_id": str(query["match_id"]),
                "candidate_match_id": str(candidate["match_id"]),
                "query_round_id": str(query["round_id"]),
                "candidate_round_id": str(candidate["round_id"]),
                "query_pov_idx": int(query["pov_idx"]),
                "candidate_pov_idx": int(candidate["pov_idx"]),
                "query_team_side": str(query.get("team_side", "")),
                "candidate_team_side": str(candidate.get("team_side", "")),
                "query_candidate_distance": float(distance),
                "positive_radius": float(positive_radius),
                "pair_policy": "same_time_spatially_colocated_pov",
                "hard_negative_policy": negative_policy,
            })
    return pd.DataFrame(pair_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None,
                        help="WebDataset shard root for the public state.bin fallback.")
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p",
                        help="Resolution of the sample_index used by the state.bin fallback.")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--candidates-per-query", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-positives", type=int, default=3)
    parser.add_argument("--positive-radius", type=float, default=1200.0)
    parser.add_argument("--negative-policy",
                        choices=[
                            "same_map_phase_different_round",
                            "same_time_far_location",
                            "same_location_wrong_time",
                        ],
                        default="same_map_phase_different_round")
    parser.add_argument("--negative-min-radius", type=float, default=None)
    parser.add_argument("--negative-location-radius", type=float, default=None,
                        help=(
                            "Maximum query-candidate distance for same_location_wrong_time "
                            "negatives. Defaults to --positive-radius."
                        ))
    parser.add_argument("--same-team-only", action="store_true")
    parser.add_argument("--include-z", action="store_true")
    parser.add_argument("--max-state-tick-gap", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = read_parquet(args.windows)
    roots = DatasetRoots.from_args(
        root=args.root,
        shard_root=args.shard_root,
        resolution=args.resolution,
    )
    pairs = build_spatial_retrieval_pairs(
        windows,
        root=args.root,
        roots=roots,
        split=args.split,
        candidates_per_query=args.candidates_per_query,
        max_queries=args.max_queries,
        max_positives=args.max_positives,
        positive_radius=args.positive_radius,
        negative_policy=args.negative_policy,
        negative_min_radius=args.negative_min_radius,
        negative_location_radius=args.negative_location_radius,
        same_team_only=args.same_team_only,
        include_z=args.include_z,
        max_tick_gap=args.max_state_tick_gap,
        seed=args.seed,
    )
    if pairs.empty:
        raise RuntimeError("no spatial retrieval pairs were produced")
    out_path = args.out / "retrieval_eval.parquet" if args.out.suffix == "" else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    positives = pairs[pairs["label"] == 1]
    negatives = pairs[pairs["label"] == 0]
    write_json(out_path.with_suffix(".metadata.json"), {
        "rows": int(len(pairs)),
        "queries": int(pairs["candidate_set_id"].nunique()),
        "positives": int(len(positives)),
        "negatives": int(len(negatives)),
        "positive_distance_mean": float(positives["query_candidate_distance"].mean()),
        "negative_distance_mean": float(negatives["query_candidate_distance"].mean()),
        "split": args.split,
        "candidates_per_query": args.candidates_per_query,
        "max_queries": args.max_queries,
        "max_positives": args.max_positives,
        "positive_radius": args.positive_radius,
        "negative_policy": args.negative_policy,
        "negative_min_radius": args.negative_min_radius,
        "negative_location_radius": args.negative_location_radius,
        "same_team_only": args.same_team_only,
        "include_z": args.include_z,
        "max_state_tick_gap": args.max_state_tick_gap,
        "seed": args.seed,
        "pairs_sha256": dataframe_sha256(pairs),
        "git_commit": git_commit(),
    })
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
