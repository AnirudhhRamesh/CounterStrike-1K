"""Report split hygiene and leakage checks for CounterStrike-1K evals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, load_release_tables, read_parquet, write_json


DENIED_COLUMNS = {
    "steam_id",
    "steamid",
    "steam64",
    "xuid",
    "account_id",
    "accountid",
    "profile_url",
    "player_name",
    "player_nickname",
    "hltv_player_id",
}


def _split_sets(df: pd.DataFrame, column: str) -> dict[str, set[str]]:
    return {
        str(split): set(group[column].dropna().astype(str).tolist())
        for split, group in df.groupby("split", sort=True)
        if column in group.columns
    }


def _pairwise_overlaps(values: dict[str, set[str]]) -> dict[str, int]:
    out = {}
    splits = sorted(values)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            out[f"{left}_x_{right}"] = len(values[left] & values[right])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--windows", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest, round_index = load_release_tables(args.root)
    match_index = read_parquet(args.root / "match_index.parquet")
    windows = read_parquet(args.windows) if args.windows is not None else None

    match_sets = _split_sets(match_index, "match_id")
    round_sets = _split_sets(round_index, "round_id")
    sample_sets = _split_sets(manifest, "sample_key")
    denied_cols = sorted(set(manifest.columns) & DENIED_COLUMNS)
    denied_cols += [c for c in manifest.columns if "steam" in c.lower() or "xuid" in c.lower()]
    denied_cols = sorted(set(denied_cols))

    by_split_map = (
        manifest.groupby(["split", "map_slug"], sort=True)
        .agg(samples=("sample_key", "count"), matches=("match_id", "nunique"), rounds=("round_id", "nunique"))
        .reset_index()
    )
    metrics = {
        "manifest_rows": int(len(manifest)),
        "match_rows": int(len(match_index)),
        "round_rows": int(len(round_index)),
        "splits": sorted(manifest["split"].dropna().astype(str).unique().tolist()),
        "match_overlap": _pairwise_overlaps(match_sets),
        "round_overlap": _pairwise_overlaps(round_sets),
        "sample_overlap": _pairwise_overlaps(sample_sets),
        "denied_public_columns_present": denied_cols,
        "split_map_rows": by_split_map.to_dict(orient="records"),
        "manifest_sha256": dataframe_sha256(manifest),
        "round_index_sha256": dataframe_sha256(round_index),
        "match_index_sha256": dataframe_sha256(match_index),
        "git_commit": git_commit(),
    }
    if windows is not None:
        window_sets = _split_sets(windows, "match_id")
        metrics.update({
            "windows_rows": int(len(windows)),
            "windows_eval_windows": int(windows["eval_window_id"].nunique()),
            "windows_match_overlap": _pairwise_overlaps(window_sets),
            "windows_sha256": dataframe_sha256(windows),
        })
    metrics["passes"] = (
        all(v == 0 for v in metrics["match_overlap"].values())
        and all(v == 0 for v in metrics["round_overlap"].values())
        and all(v == 0 for v in metrics["sample_overlap"].values())
        and not denied_cols
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, metrics)
    by_split_map.to_csv(args.out.with_suffix(".split_map.csv"), index=False)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
