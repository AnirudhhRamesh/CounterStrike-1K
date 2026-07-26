"""Evaluate a strict retrieval sweep with tie-aware, query-paired statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.embeddings import load_embedding_table
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.stats import cluster_bootstrap_column_means
from cs2_release.retrieval.eval_basic import attach_embedding_ids
from cs2_release.retrieval.protocol_sweep import validate_strict_pairs

QUERY_METRICS = ["top1", "top5", "mrr", "positive_rank"]


def _expected_reciprocal_rank(greater: int, tied: int) -> float:
    return float(np.mean([1.0 / rank for rank in range(greater + 1, greater + tied + 1)]))


def _tie_hit_probability(greater: np.ndarray, tied: np.ndarray, k: int) -> np.ndarray:
    remaining = np.clip(float(k) - greater.astype(np.float64), 0.0, tied.astype(np.float64))
    return remaining / tied.astype(np.float64)


def score_pairs(
    pairs: pd.DataFrame,
    index: pd.DataFrame,
    normalized_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = int(pairs["declared_candidates"].iloc[0])
    validate_strict_pairs(pairs, candidates_per_query=candidates)
    attached = attach_embedding_ids(pairs, index)
    if len(attached) != len(pairs):
        raise ValueError(
            f"embedding join dropped {len(pairs) - len(attached)} of {len(pairs)} pair rows"
        )
    query_ids = attached["query_embedding_row_id"].to_numpy(dtype=np.int64)
    candidate_ids = attached["candidate_embedding_row_id"].to_numpy(dtype=np.int64)
    if not (
        np.isfinite(normalized_embeddings[query_ids]).all()
        and np.isfinite(normalized_embeddings[candidate_ids]).all()
    ):
        raise ValueError("non-finite embeddings are referenced by a strict sweep cell")
    predictions = attached.copy()
    predictions["score"] = np.sum(
        normalized_embeddings[query_ids] * normalized_embeddings[candidate_ids],
        axis=1,
    ).astype(np.float32)

    positives = predictions.loc[
        predictions["label"] == 1,
        ["candidate_set_id", "score"],
    ].rename(columns={"score": "positive_score"})
    scored = predictions.merge(positives, on="candidate_set_id", validate="many_to_one")
    scored["greater_than_positive"] = scored["score"] > scored["positive_score"]
    scored["tied_with_positive"] = scored["score"] == scored["positive_score"]
    tie_counts = scored.groupby("candidate_set_id", sort=False).agg(
        greater=("greater_than_positive", "sum"),
        tied=("tied_with_positive", "sum"),
    )
    if (tie_counts["tied"] < 1).any():
        raise ValueError("positive candidate is absent from its own score-tie group")

    first = (
        predictions.sort_values(["candidate_set_id", "candidate_ordinal"])
        .groupby("candidate_set_id", sort=False)
        .first()
    )
    summary = first[[
        "sweep_query_id",
        "query_match_id",
        "map_slug",
        "hard_negative_policy",
        "pair_sampling_seed",
        "declared_candidates",
    ]].join(tie_counts)
    greater = summary["greater"].to_numpy(dtype=np.int64)
    tied = summary["tied"].to_numpy(dtype=np.int64)
    summary["top1"] = _tie_hit_probability(greater, tied, 1)
    summary["top5"] = _tie_hit_probability(greater, tied, 5)
    summary["mrr"] = [
        _expected_reciprocal_rank(int(g), int(t))
        for g, t in zip(greater, tied, strict=True)
    ]
    summary["positive_rank"] = greater + (tied + 1.0) / 2.0
    summary["chance_top1"] = 1.0 / candidates
    summary["chance_top5"] = min(5, candidates) / candidates
    summary["top1_lift_over_chance"] = (
        summary["top1"] - summary["chance_top1"]
    ) / (1.0 - summary["chance_top1"])
    summary = summary.reset_index()
    return predictions, summary


def _metric_record(query_metrics: pd.DataFrame) -> dict[str, float | int]:
    candidates = int(query_metrics["declared_candidates"].iloc[0])
    record: dict[str, float | int] = {
        "queries": len(query_metrics),
        "matches": int(query_metrics["query_match_id"].nunique()),
        "maps": int(query_metrics["map_slug"].nunique()),
        "candidate_count": candidates,
        "chance_top1": 1.0 / candidates,
        "chance_top5": min(5, candidates) / candidates,
        "tie_queries": int((query_metrics["tied"] > 1).sum()),
    }
    for metric in [*QUERY_METRICS, "top1_lift_over_chance"]:
        record[metric] = float(query_metrics[metric].mean())
    return record


def _aggregate_cells(
    query_metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict]]:
    aggregate_records = []
    per_map_records = []
    group_columns = ["hard_negative_policy", "declared_candidates"]
    for (policy, candidates), group in query_metrics.groupby(group_columns, sort=True):
        averaged = (
            group.groupby(
                ["sweep_query_id", "query_match_id", "map_slug"],
                as_index=False,
                sort=False,
            )[[*QUERY_METRICS, "top1_lift_over_chance"]]
            .mean()
        )
        record = {
            "hard_negative_policy": str(policy),
            "candidate_count": int(candidates),
            "pair_sampling_seeds": int(group["pair_sampling_seed"].nunique()),
            **_metric_record(group),
            "queries": len(averaged),
            "matches": int(averaged["query_match_id"].nunique()),
            "maps": int(averaged["map_slug"].nunique()),
        }
        record.update(
            cluster_bootstrap_column_means(
                averaged,
                cluster_col="query_match_id",
                metrics=QUERY_METRICS,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )
        aggregate_records.append(record)
        for map_slug, map_group in averaged.groupby("map_slug", sort=True):
            per_map_records.append({
                "hard_negative_policy": str(policy),
                "candidate_count": int(candidates),
                "map_slug": str(map_slug),
                "queries": len(map_group),
                **{
                    metric: float(map_group[metric].mean())
                    for metric in QUERY_METRICS
                },
            })
    return aggregate_records, per_map_records


def _paired_policy_contrasts(
    query_metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    contrasts = []
    policies = sorted(query_metrics["hard_negative_policy"].unique())
    for candidates, count_group in query_metrics.groupby("declared_candidates", sort=True):
        averaged = (
            count_group.groupby(
                [
                    "sweep_query_id",
                    "query_match_id",
                    "hard_negative_policy",
                ],
                as_index=False,
                sort=False,
            )[QUERY_METRICS]
            .mean()
        )
        for left_idx, left in enumerate(policies):
            for right in policies[left_idx + 1:]:
                pair = averaged[
                    averaged["hard_negative_policy"].isin([left, right])
                ]
                if pair["hard_negative_policy"].nunique() != 2:
                    continue
                metadata = pair[["sweep_query_id", "query_match_id"]].drop_duplicates(
                    "sweep_query_id"
                )
                wide = pair.pivot(
                    index="sweep_query_id",
                    columns="hard_negative_policy",
                    values=QUERY_METRICS,
                )
                if left not in wide.columns.levels[1] or right not in wide.columns.levels[1]:
                    continue
                complete = wide.dropna()
                if complete.empty:
                    continue
                delta = metadata.set_index("sweep_query_id").loc[complete.index].copy()
                for metric in QUERY_METRICS:
                    delta[metric] = complete[(metric, right)] - complete[(metric, left)]
                record = {
                    "candidate_count": int(candidates),
                    "left_policy": str(left),
                    "right_policy": str(right),
                    "direction": "right_minus_left",
                    "paired_queries": len(delta),
                }
                for metric in QUERY_METRICS:
                    record[metric] = float(delta[metric].mean())
                record.update(
                    cluster_bootstrap_column_means(
                        delta.reset_index(),
                        cluster_col="query_match_id",
                        metrics=QUERY_METRICS,
                        n_boot=bootstrap_samples,
                        seed=bootstrap_seed + 101,
                    )
                )
                contrasts.append(record)
    return contrasts


def evaluate_sweep(
    *,
    pairs_root: Path,
    embeddings_root: Path,
    output_root: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    contract = json.loads((pairs_root / "protocol_sweep.json").read_text(encoding="utf-8"))
    if contract.get("status") != "pass":
        raise ValueError("protocol sweep contract did not pass")
    index, embeddings = load_embedding_table(embeddings_root)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_root = output_root / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    cell_records = []
    all_query_metrics = []
    for pair_path in sorted(pairs_root.glob("*.parquet")):
        cell_name = pair_path.stem
        if cell_name not in contract["cells"]:
            raise ValueError(f"{pair_path}: missing cell metadata")
        pairs = read_parquet(pair_path)
        expected_hash = contract["cells"][cell_name]["pairs_sha256"]
        actual_hash = dataframe_sha256(pairs)
        if actual_hash != expected_hash:
            raise ValueError(f"{pair_path}: pair hash mismatch")
        predictions, query_metrics = score_pairs(pairs, index, normalized)
        predictions.to_parquet(predictions_root / pair_path.name, index=False)
        query_metrics["cell_name"] = cell_name
        all_query_metrics.append(query_metrics)
        cell_records.append({
            "cell_name": cell_name,
            "pair_sampling_seed": int(query_metrics["pair_sampling_seed"].iloc[0]),
            "hard_negative_policy": str(query_metrics["hard_negative_policy"].iloc[0]),
            **_metric_record(query_metrics),
        })

    query_table = pd.concat(all_query_metrics, ignore_index=True)
    aggregate, per_map = _aggregate_cells(
        query_table,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    contrasts = _paired_policy_contrasts(
        query_table,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    pd.DataFrame(cell_records).to_parquet(output_root / "cell_metrics.parquet", index=False)
    query_table.to_parquet(output_root / "query_metrics.parquet", index=False)
    pd.DataFrame(aggregate).to_parquet(output_root / "aggregate_metrics.parquet", index=False)
    pd.DataFrame(per_map).to_parquet(output_root / "per_map_metrics.parquet", index=False)
    pd.DataFrame(contrasts).to_parquet(output_root / "paired_policy_contrasts.parquet", index=False)
    result = {
        "schema": "cs2-retrieval-protocol-sweep-evaluation-v1",
        "status": "pass",
        "pairs_contract": str(pairs_root / "protocol_sweep.json"),
        "pairs_cohort_sha256": contract["cohort_sha256"],
        "embedding_rows": len(index),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_index_sha256": dataframe_sha256(index.drop(columns=["embedding_row_id"])),
        "cells": cell_records,
        "aggregate": aggregate,
        "per_map": per_map,
        "paired_policy_contrasts": contrasts,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "query_metrics_sha256": dataframe_sha256(query_table),
        "git_commit": git_commit(),
    }
    write_json(output_root / "evaluation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    result = evaluate_sweep(
        pairs_root=args.pairs_root,
        embeddings_root=args.embeddings,
        output_root=args.out,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if result["status"] != "pass" or not math.isfinite(float(result["embedding_dim"])):
        raise RuntimeError("sweep evaluation did not complete")
    print(args.out / "evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
