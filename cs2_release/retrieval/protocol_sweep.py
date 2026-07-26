"""Build fail-closed retrieval protocol sweeps for shortcut sensitivity.

Unlike the original submission-time builders, this module never samples a
candidate twice, never relaxes a requested negative policy, and never changes
the positive POV when only the candidate count changes.  Sweep cells are
restricted to their shared query cohort before they are written.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json

NEGATIVE_POLICIES = (
    "same_map_phase_different_round",
    "same_match_wrong_round",
    "same_round_wrong_time",
    "same_target_pov_same_round_wrong_time",
)


def _stable_int(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _candidate_identity(row: pd.Series) -> tuple[str, int]:
    return str(row["eval_window_id"]), int(row["pov_idx"])


def _query_identity(row: pd.Series) -> str:
    return f"{row['eval_window_id']}__q{int(row['pov_idx']):02d}"


def _choose_positive(
    query: pd.Series,
    same_window: pd.DataFrame,
    *,
    seed: int,
) -> pd.Series | None:
    positives = same_window[same_window["pov_idx"] != int(query["pov_idx"])].copy()
    if positives.empty:
        return None
    positives["__order"] = [
        _stable_int(seed, "positive", _query_identity(query), *_candidate_identity(row))
        for _, row in positives.iterrows()
    ]
    return positives.sort_values(["__order", "pov_idx"]).iloc[0]


def _index_rows(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[tuple[object, ...], list[int]]:
    result: dict[tuple[object, ...], list[int]] = {}
    for key, group in df.groupby(columns, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        result[tuple(key)] = group.index.astype(int).tolist()
    return result


def _coprime_step(length: int, raw_step: int) -> int:
    if length <= 1:
        return 1
    step = 1 + raw_step % (length - 1)
    while math.gcd(step, length) != 1:
        step = 1 + step % (length - 1)
    return step


def _negative_indices(
    df: pd.DataFrame,
    query: pd.Series,
    positive: pd.Series,
    *,
    policy: str,
    indexes: Mapping[str, dict[tuple[object, ...], list[int]]],
    needed: int,
    seed: int,
) -> list[int]:
    if policy == "same_map_phase_different_round":
        key = (str(query["map_slug"]), str(query["phase_bucket"]))
        pool = indexes["map_phase"].get(key, [])
    elif policy == "same_match_wrong_round":
        key = (str(query["match_id"]), str(query["phase_bucket"]))
        pool = indexes["match_phase"].get(key, [])
    elif policy == "same_round_wrong_time":
        key = (str(query["round_id"]),)
        pool = indexes["round"].get(key, [])
    elif policy == "same_target_pov_same_round_wrong_time":
        key = (str(query["round_id"]), int(positive["pov_idx"]))
        pool = indexes["round_pov"].get(key, [])
    else:
        raise ValueError(f"unknown negative policy {policy!r}; expected one of {NEGATIVE_POLICIES}")
    if not pool:
        return []

    query_round = str(query["round_id"])
    query_window = str(query["eval_window_id"])
    forbidden = {
        int(query["window_row_id"]),
        int(positive["window_row_id"]),
    }
    start_hash = _stable_int(seed, "negative-start", policy, _query_identity(query))
    step_hash = _stable_int(seed, "negative-step", policy, _query_identity(query))
    start = start_hash % len(pool)
    step = _coprime_step(len(pool), step_hash)
    selected: list[int] = []
    for offset in range(len(pool)):
        row_idx = pool[(start + offset * step) % len(pool)]
        if row_idx in forbidden:
            continue
        candidate = df.iloc[row_idx]
        if policy in {
            "same_map_phase_different_round",
            "same_match_wrong_round",
        }:
            if str(candidate["round_id"]) == query_round:
                continue
        else:
            if str(candidate["eval_window_id"]) == query_window:
                continue
        selected.append(row_idx)
        if len(selected) == needed:
            return selected
    return []


def _query_plan(
    df: pd.DataFrame,
    *,
    max_queries: int | None,
    seed: int,
) -> pd.DataFrame:
    queries = df.sort_values(
        ["map_slug", "match_id", "round_idx", "window_idx", "pov_idx"]
    ).copy()
    queries["__query_order"] = [
        _stable_int(seed, "query", _query_identity(row))
        for _, row in queries.iterrows()
    ]
    queries = queries.sort_values(["__query_order", "eval_window_id", "pov_idx"])
    if max_queries is not None:
        queries = queries.head(max_queries)
    return queries


def validate_strict_pairs(
    pairs: pd.DataFrame,
    *,
    candidates_per_query: int,
) -> dict[str, int]:
    """Raise on leakage or cardinality drift and return compact diagnostics."""

    if pairs.empty:
        raise ValueError("retrieval pair table is empty")
    grouped = pairs.groupby("candidate_set_id", sort=False)
    sizes = grouped.size()
    if not (sizes == candidates_per_query).all():
        raise ValueError(
            "candidate count drift: "
            f"expected {candidates_per_query}, observed {sorted(sizes.unique().tolist())}"
        )
    positives = grouped["label"].sum()
    if not (positives == 1).all():
        raise ValueError(
            f"each candidate set must have exactly one positive; observed {sorted(positives.unique())}"
        )
    duplicate_counts = grouped["candidate_window_row_id"].nunique()
    if not (duplicate_counts == candidates_per_query).all():
        raise ValueError("duplicate candidates detected within a candidate set")
    self_rows = pairs[
        (pairs["query_eval_window_id"] == pairs["candidate_eval_window_id"])
        & (pairs["query_pov_idx"] == pairs["candidate_pov_idx"])
    ]
    if not self_rows.empty:
        raise ValueError("query POV leaked into its own candidate set")
    if pairs["candidate_ordinal"].min() != 0 or pairs["candidate_ordinal"].max() >= candidates_per_query:
        raise ValueError("candidate ordinal is outside the declared candidate count")
    return {
        "queries": int(pairs["candidate_set_id"].nunique()),
        "rows": int(len(pairs)),
        "candidates_per_query": int(candidates_per_query),
        "positives": int(pairs["label"].sum()),
        "duplicate_candidate_rows": 0,
        "self_candidate_rows": 0,
    }


def build_strict_retrieval_pairs(
    windows: pd.DataFrame,
    *,
    split: str,
    candidates_per_query: int,
    max_queries: int | None,
    negative_policy: str,
    seed: int,
    query_seed: int = 123,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    if candidates_per_query < 2:
        raise ValueError("candidates_per_query must be at least 2")
    if negative_policy not in NEGATIVE_POLICIES:
        raise ValueError(f"unknown negative policy {negative_policy!r}")

    df = windows[windows["split"] == split].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no windows for split {split!r}")
    duplicate_windows = df.duplicated(["eval_window_id", "pov_idx"], keep=False)
    if duplicate_windows.any():
        raise ValueError("windows contain duplicate (eval_window_id, pov_idx) rows")
    df["window_row_id"] = np.arange(len(df), dtype=np.int64)
    groups = {str(key): group for key, group in df.groupby("eval_window_id", sort=False)}
    indexes = {
        "map_phase": _index_rows(df, ["map_slug", "phase_bucket"]),
        "match_phase": _index_rows(df, ["match_id", "phase_bucket"]),
        "round": _index_rows(df, ["round_id"]),
        "round_pov": _index_rows(df, ["round_id", "pov_idx"]),
    }
    queries = _query_plan(df, max_queries=max_queries, seed=query_seed)

    rows: list[dict] = []
    skipped_no_positive = 0
    skipped_insufficient_negatives = 0
    for _, query in queries.iterrows():
        positive = _choose_positive(
            query,
            groups[str(query["eval_window_id"])],
            seed=seed,
        )
        if positive is None:
            skipped_no_positive += 1
            continue
        needed = candidates_per_query - 1
        negative_indices = _negative_indices(
            df,
            query,
            positive,
            policy=negative_policy,
            indexes=indexes,
            needed=needed,
            seed=seed,
        )
        if len(negative_indices) < needed:
            skipped_insufficient_negatives += 1
            continue
        negatives = df.iloc[negative_indices]
        candidate_set_id = (
            f"{_query_identity(query)}__t{int(positive['pov_idx']):02d}"
        )
        candidates = [(positive, 1), *((row, 0) for _, row in negatives.iterrows())]
        candidates.sort(
            key=lambda item: _stable_int(
                seed,
                "shuffle",
                negative_policy,
                candidate_set_id,
                *_candidate_identity(item[0]),
            )
        )
        for ordinal, (candidate, label) in enumerate(candidates):
            rows.append({
                "candidate_set_id": candidate_set_id,
                "sweep_query_id": _query_identity(query),
                "query_window_row_id": int(query["window_row_id"]),
                "candidate_window_row_id": int(candidate["window_row_id"]),
                "label": int(label),
                "candidate_ordinal": int(ordinal),
                "declared_candidates": int(candidates_per_query),
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
                "positive_target_pov_idx": int(positive["pov_idx"]),
                "pair_policy": "same_time_one_positive_pov",
                "hard_negative_policy": negative_policy,
                "pair_sampling_seed": int(seed),
            })
    pairs = pd.DataFrame(rows)
    diagnostics: dict[str, int | str] = {
        "split": split,
        "negative_policy": negative_policy,
        "seed": int(seed),
        "query_seed": int(query_seed),
        "requested_queries": int(len(queries)),
        "skipped_no_positive": int(skipped_no_positive),
        "skipped_insufficient_unique_negatives": int(skipped_insufficient_negatives),
    }
    if not pairs.empty:
        diagnostics.update(
            validate_strict_pairs(pairs, candidates_per_query=candidates_per_query)
        )
    return pairs, diagnostics


def align_sweep_query_cohort(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if not tables:
        raise ValueError("at least one sweep table is required")
    cohort_sets = []
    for name, table in tables.items():
        if table.empty:
            raise ValueError(f"sweep cell {name!r} has no eligible queries")
        cohort_sets.append(set(table["sweep_query_id"].astype(str).unique()))
    cohort = sorted(set.intersection(*cohort_sets))
    if not cohort:
        raise ValueError("sweep cells have no shared eligible query cohort")
    cohort_set = set(cohort)
    aligned = {
        name: table[table["sweep_query_id"].astype(str).isin(cohort_set)]
        .sort_values(["candidate_set_id", "candidate_ordinal"])
        .reset_index(drop=True)
        for name, table in tables.items()
    }
    return aligned, cohort


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _parse_csv_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _cell_name(policy: str, candidates: int, seed: int) -> str:
    return f"{policy}__c{candidates:03d}__s{seed:05d}"


def build_sweep(
    windows: pd.DataFrame,
    *,
    split: str,
    candidate_counts: Iterable[int],
    negative_policies: Iterable[str],
    seeds: Iterable[int],
    max_queries: int | None,
    query_seed: int = 123,
) -> tuple[dict[str, pd.DataFrame], dict]:
    raw: dict[str, pd.DataFrame] = {}
    cell_diagnostics: dict[str, dict] = {}
    settings: dict[str, tuple[str, int, int]] = {}
    for seed in seeds:
        for policy in negative_policies:
            for candidates in candidate_counts:
                name = _cell_name(policy, candidates, seed)
                table, diagnostics = build_strict_retrieval_pairs(
                    windows,
                    split=split,
                    candidates_per_query=int(candidates),
                    max_queries=max_queries,
                    negative_policy=policy,
                    seed=int(seed),
                    query_seed=int(query_seed),
                )
                raw[name] = table
                cell_diagnostics[name] = diagnostics
                settings[name] = (policy, int(candidates), int(seed))
    aligned, cohort = align_sweep_query_cohort(raw)
    for name, table in aligned.items():
        _, candidates, _ = settings[name]
        validate_strict_pairs(table, candidates_per_query=candidates)
        cell_diagnostics[name]["aligned_queries"] = int(len(cohort))
        cell_diagnostics[name]["aligned_rows"] = int(len(table))
        cell_diagnostics[name]["pairs_sha256"] = dataframe_sha256(table)
    metadata = {
        "schema": "cs2-retrieval-protocol-sweep-v1",
        "status": "pass",
        "split": split,
        "candidate_counts": [int(value) for value in candidate_counts],
        "negative_policies": list(negative_policies),
        "seeds": [int(value) for value in seeds],
        "max_queries": max_queries,
        "query_seed": int(query_seed),
        "aligned_queries": int(len(cohort)),
        "cohort_sha256": hashlib.sha256("\n".join(cohort).encode("utf-8")).hexdigest(),
        "cells": cell_diagnostics,
        "git_commit": git_commit(),
    }
    return aligned, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--candidate-counts", type=_parse_csv_ints, default=[4, 8, 16, 32])
    parser.add_argument(
        "--negative-policies",
        type=_parse_csv_strings,
        default=list(NEGATIVE_POLICIES),
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=[17, 29, 43, 71, 101])
    parser.add_argument("--max-queries", type=int, default=7500)
    parser.add_argument(
        "--query-seed",
        type=int,
        default=123,
        help="Fixed query-subsampling seed shared by every pairing seed and sweep cell.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.negative_policies) - set(NEGATIVE_POLICIES))
    if unknown:
        parser.error(f"unknown negative policies: {unknown}")
    windows = read_parquet(args.windows)
    tables, metadata = build_sweep(
        windows,
        split=args.split,
        candidate_counts=args.candidate_counts,
        negative_policies=args.negative_policies,
        seeds=args.seeds,
        max_queries=args.max_queries,
        query_seed=args.query_seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(args.out / f"{name}.parquet", index=False)
    write_json(args.out / "protocol_sweep.json", metadata)
    print(args.out / "protocol_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
