"""Build hard-negative cross-POV retrieval candidate sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json


def build_retrieval_pairs(
    windows: pd.DataFrame,
    *,
    split: str,
    candidates_per_query: int,
    max_queries: int | None,
    negative_policy: str = "same_map_phase_different_round",
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = windows[windows["split"] == split].copy().reset_index(drop=True)
    df["window_row_id"] = np.arange(len(df), dtype=np.int64)
    groups = {str(k): g.copy() for k, g in df.groupby("eval_window_id", sort=False)}
    query_rows = df.sort_values(["map_slug", "phase_bucket", "round_id", "pov_idx"]).copy()
    if max_queries is not None and len(query_rows) > max_queries:
        idx = np.sort(rng.choice(query_rows.index.to_numpy(), size=max_queries, replace=False))
        query_rows = query_rows.loc[idx].sort_values(["map_slug", "phase_bucket", "round_id", "pov_idx"])

    pair_rows: list[dict] = []
    for _, query in query_rows.iterrows():
        same_window = groups[str(query["eval_window_id"])]
        positives = same_window[same_window["pov_idx"] != int(query["pov_idx"])]
        if positives.empty:
            continue
        positive = positives.iloc[int(rng.integers(0, len(positives)))]
        if negative_policy == "same_map_phase_different_round":
            neg_pool = df[
                (df["map_slug"] == query["map_slug"])
                & (df["phase_bucket"] == query["phase_bucket"])
                & (df["round_id"] != query["round_id"])
                & (df["pov_idx"] != int(query["pov_idx"]))
            ]
        elif negative_policy == "same_match_wrong_round":
            neg_pool = df[
                (df["match_id"] == query["match_id"])
                & (df["phase_bucket"] == query["phase_bucket"])
                & (df["round_id"] != query["round_id"])
                & (df["pov_idx"] != int(query["pov_idx"]))
            ]
        elif negative_policy == "same_round_wrong_time":
            neg_pool = df[
                (df["round_id"] == query["round_id"])
                & (df["eval_window_id"] != query["eval_window_id"])
                & (df["pov_idx"] != int(query["pov_idx"]))
            ]
        else:
            raise ValueError(f"unknown negative_policy={negative_policy!r}")
        if neg_pool.empty:
            neg_pool = df[
                (df["map_slug"] == query["map_slug"])
                & (df["round_id"] != query["round_id"])
                & (df["pov_idx"] != int(query["pov_idx"]))
            ]
        if neg_pool.empty:
            continue
        n_neg = max(1, candidates_per_query - 1)
        replace = len(neg_pool) < n_neg
        neg_indices = rng.choice(neg_pool.index.to_numpy(), size=n_neg, replace=replace)
        candidate_set_id = f"{query['eval_window_id']}__q{int(query['pov_idx']):02d}"
        candidates = [(positive, 1)]
        candidates.extend((df.loc[idx], 0) for idx in neg_indices)
        for rank_idx, (candidate, label) in enumerate(candidates):
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
                "query_window_idx": int(query.get("window_idx", 0)),
                "candidate_window_idx": int(candidate.get("window_idx", 0)),
                "query_start_tick": int(query.get("start_tick", 0)),
                "candidate_start_tick": int(candidate.get("start_tick", 0)),
                "hard_negative_policy": negative_policy,
            })
    return pd.DataFrame(pair_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--candidates-per-query", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--negative-policy",
        choices=[
            "same_map_phase_different_round",
            "same_match_wrong_round",
            "same_round_wrong_time",
        ],
        default="same_map_phase_different_round",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = read_parquet(args.windows)
    pairs = build_retrieval_pairs(
        windows,
        split=args.split,
        candidates_per_query=args.candidates_per_query,
        max_queries=args.max_queries,
        negative_policy=args.negative_policy,
        seed=args.seed,
    )
    if pairs.empty:
        raise RuntimeError("no retrieval pairs were produced")
    out_path = args.out / "retrieval_eval.parquet" if args.out.suffix == "" else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    write_json(out_path.with_suffix(".metadata.json"), {
        "rows": int(len(pairs)),
        "queries": int(pairs["candidate_set_id"].nunique()),
        "split": args.split,
        "candidates_per_query": args.candidates_per_query,
        "max_queries": args.max_queries,
        "negative_policy": args.negative_policy,
        "seed": args.seed,
        "pairs_sha256": dataframe_sha256(pairs),
        "git_commit": git_commit(),
    })
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
