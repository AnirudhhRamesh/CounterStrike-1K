"""Build a strict 3D co-location-radius retrieval sweep."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from cs2_release.core.io import (
    DatasetRoots,
    dataframe_sha256,
    git_commit,
    read_parquet,
    write_json,
)
from cs2_release.retrieval.pairs.spatial import (
    POSITION_COLUMNS,
    attach_midpoint_positions,
    build_spatial_retrieval_pairs,
)


def _parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated radii")
    return values


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated seeds")
    return values


def validate_spatial_cell(
    pairs: pd.DataFrame,
    *,
    radius: float,
    candidates_per_query: int,
) -> dict:
    if pairs.empty:
        raise ValueError("spatial sweep cell is empty")
    groups = pairs.groupby("candidate_set_id", sort=False)
    sizes = groups.size()
    if not (sizes == candidates_per_query).all():
        raise ValueError("spatial candidate-count drift")
    positives = groups["label"].sum()
    if not (positives == 1).all():
        raise ValueError("spatial sweep requires exactly one positive per query")
    unique = groups["candidate_window_row_id"].nunique()
    if not (unique == candidates_per_query).all():
        raise ValueError("duplicate candidate in spatial sweep")
    positive_rows = pairs[pairs["label"] == 1]
    negative_rows = pairs[pairs["label"] == 0]
    tolerance = 1e-4
    if (positive_rows["query_candidate_distance"] > radius + tolerance).any():
        raise ValueError("out-of-radius positive in spatial sweep")
    if (negative_rows["query_candidate_distance"] > radius + tolerance).any():
        raise ValueError("out-of-radius same-location negative in spatial sweep")
    if (negative_rows["query_round_id"] == negative_rows["candidate_round_id"]).any():
        raise ValueError("same-location wrong-time negative came from the query round")
    return {
        "queries": int(pairs["candidate_set_id"].nunique()),
        "rows": len(pairs),
        "positives": len(positive_rows),
        "negatives": len(negative_rows),
        "positive_distance_mean": float(positive_rows["query_candidate_distance"].mean()),
        "negative_distance_mean": float(negative_rows["query_candidate_distance"].mean()),
    }


def build_spatial_sweep(
    positioned_windows: pd.DataFrame,
    *,
    root: Path,
    split: str,
    radii: list[float],
    seeds: list[int],
    candidates_per_query: int,
    max_queries: int | None,
    query_seed: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    raw = {}
    diagnostics = {}
    for seed in seeds:
        for radius in radii:
            name = f"radius_{radius:g}__c{candidates_per_query:03d}__s{seed:05d}"
            pairs = build_spatial_retrieval_pairs(
                positioned_windows,
                root=root,
                roots=None,
                split=split,
                candidates_per_query=candidates_per_query,
                max_queries=max_queries,
                max_positives=1,
                positive_radius=radius,
                negative_policy="same_location_wrong_time",
                negative_min_radius=None,
                negative_location_radius=radius,
                same_team_only=False,
                include_z=True,
                max_tick_gap=16,
                seed=seed,
                query_seed=query_seed,
            )
            raw[name] = pairs
            diagnostics[name] = {
                "radius": float(radius),
                "seed": int(seed),
                "pre_alignment_queries": int(pairs["candidate_set_id"].nunique())
                if not pairs.empty else 0,
            }
    if any(table.empty for table in raw.values()):
        empty = [name for name, table in raw.items() if table.empty]
        raise ValueError(f"empty spatial sweep cells: {empty}")
    cohort = sorted(set.intersection(*[
        set(table["candidate_set_id"].astype(str).unique())
        for table in raw.values()
    ]))
    if not cohort:
        raise ValueError("spatial radius cells have no shared query cohort")
    cohort_set = set(cohort)
    aligned = {}
    for name, table in raw.items():
        radius = float(diagnostics[name]["radius"])
        table = table[table["candidate_set_id"].astype(str).isin(cohort_set)].copy()
        table["declared_candidates"] = int(candidates_per_query)
        table["spatial_radius"] = radius
        table["pair_sampling_seed"] = int(diagnostics[name]["seed"])
        table["sweep_query_id"] = table["candidate_set_id"].astype(str)
        table = table.sort_values(["candidate_set_id", "candidate_ordinal"]).reset_index(drop=True)
        diagnostics[name].update(
            validate_spatial_cell(
                table,
                radius=radius,
                candidates_per_query=candidates_per_query,
            )
        )
        diagnostics[name]["pairs_sha256"] = dataframe_sha256(table)
        aligned[name] = table
    metadata = {
        "schema": "cs2-spatial-retrieval-protocol-sweep-v1",
        "status": "pass",
        "split": split,
        "radii": [float(value) for value in radii],
        "seeds": [int(value) for value in seeds],
        "candidates_per_query": int(candidates_per_query),
        "max_queries": max_queries,
        "query_seed": int(query_seed),
        "include_z": True,
        "negative_policy": "same_location_wrong_time",
        "aligned_queries": len(cohort),
        "cohort_sha256": hashlib.sha256("\n".join(cohort).encode()).hexdigest(),
        "cells": diagnostics,
        "positioned_windows_sha256": dataframe_sha256(positioned_windows),
        "git_commit": git_commit(),
    }
    return aligned, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--radii", type=_parse_csv_floats, default=[400, 800, 1200, 1600, 2400])
    parser.add_argument("--seeds", type=_parse_csv_ints, default=[17, 29, 43, 71, 101])
    parser.add_argument("--candidates-per-query", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=2500)
    parser.add_argument("--query-seed", type=int, default=123)
    parser.add_argument("--max-state-tick-gap", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.candidates_per_query < 2:
        parser.error("--candidates-per-query must be at least 2")
    roots = DatasetRoots.from_args(
        root=args.root,
        shard_root=args.shard_root,
        resolution=args.resolution,
    )
    windows = read_parquet(args.windows)
    positioned = attach_midpoint_positions(
        windows[windows["split"] == args.split].copy(),
        root=args.root,
        roots=roots,
        max_tick_gap=args.max_state_tick_gap,
    )
    if positioned.empty or not set(POSITION_COLUMNS).issubset(positioned.columns):
        raise RuntimeError("no positioned windows were produced")
    args.out.mkdir(parents=True, exist_ok=True)
    positioned.to_parquet(args.out / "positioned_windows.parquet", index=False)
    tables, metadata = build_spatial_sweep(
        positioned,
        root=args.root,
        split=args.split,
        radii=args.radii,
        seeds=args.seeds,
        candidates_per_query=args.candidates_per_query,
        max_queries=args.max_queries,
        query_seed=args.query_seed,
    )
    for name, table in tables.items():
        table.to_parquet(args.out / f"{name}.parquet", index=False)
    write_json(args.out / "protocol_sweep.json", metadata)
    print(args.out / "protocol_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
