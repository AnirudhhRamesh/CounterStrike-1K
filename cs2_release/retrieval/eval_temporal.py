"""Diagnose temporal alignment in same-round cross-POV retrieval outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import binary_auc, binary_average_precision
from cs2_release.core.stats import cluster_bootstrap_column_means


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def add_temporal_offsets(predictions: pd.DataFrame, *, tick_rate: float) -> pd.DataFrame:
    out = predictions.copy()
    query_tick_col = _first_present(out, ["query_start_tick", "query_mid_tick"])
    candidate_tick_col = _first_present(out, ["candidate_start_tick", "candidate_mid_tick"])
    if query_tick_col is None or candidate_tick_col is None:
        raise ValueError(
            "predictions must include query/candidate temporal columns such as "
            "query_start_tick and candidate_start_tick"
        )
    out["temporal_offset_ticks"] = (
        out[candidate_tick_col].astype(np.float64) - out[query_tick_col].astype(np.float64)
    ).abs()
    out["temporal_offset_s"] = out["temporal_offset_ticks"] / float(tick_rate)
    return out


def _bucket_labels(edges: list[float]) -> list[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi >= 1e8:
            labels.append(f">={lo:g}s")
        elif lo == 0:
            labels.append(f"<{hi:g}s")
        else:
            labels.append(f"{lo:g}-{hi:g}s")
    return labels


def bucket_temporal_offsets(df: pd.DataFrame, *, edges: list[float]) -> pd.DataFrame:
    out = df.copy()
    labels = _bucket_labels(edges)
    out["temporal_bucket"] = pd.cut(
        out["temporal_offset_s"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)
    out.loc[out["label"].astype(int) == 1, "temporal_bucket"] = "synchronized"
    return out


def query_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_set_id, group in predictions.groupby("candidate_set_id", sort=False):
        ranked = group.sort_values("rank")
        pos = group[group["label"].astype(int) == 1]
        neg = group[group["label"].astype(int) == 0]
        if pos.empty or neg.empty:
            continue
        pos_score = float(pos["score"].max())
        top = ranked.iloc[0]
        rows.append({
            "candidate_set_id": str(candidate_set_id),
            "query_match_id": str(group.get("query_match_id", group.get("match_id", "")).iloc[0])
            if ("query_match_id" in group.columns or "match_id" in group.columns)
            else str(group["query_eval_window_id"].iloc[0]).split("__", 1)[0],
            "hit@1": float(int(top["label"]) == 1),
            "hit@5": float((ranked.head(5)["label"].astype(int) == 1).any()),
            "positive_score": pos_score,
            "top_negative_score": float(neg["score"].max()),
            "positive_minus_top_negative": pos_score - float(neg["score"].max()),
            "top_negative_offset_s": float(neg.sort_values("score", ascending=False).iloc[0]["temporal_offset_s"]),
        })
    return pd.DataFrame(rows)


def bucket_summary(predictions: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows = []
    for bucket, group in predictions.groupby("temporal_bucket", sort=False):
        values = group["score"].to_numpy(dtype=np.float64)
        rows.append({
            "temporal_bucket": str(bucket),
            "rows": int(len(group)),
            "candidate_sets": int(group["candidate_set_id"].nunique()),
            "label_mean": float(group["label"].astype(float).mean()),
            "offset_s_mean": float(group["temporal_offset_s"].mean()),
            "score_mean": float(np.mean(values)) if len(values) else float("nan"),
            "score_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    out = pd.DataFrame(rows)
    if out.empty or bootstrap_samples <= 0:
        return out
    ci_rows = []
    for bucket, group in predictions.groupby("temporal_bucket", sort=False):
        summary = group[["query_match_id", "score"]].copy()
        bucket_seed = int(hashlib.sha256(str(bucket).encode("utf-8")).hexdigest()[:8], 16)
        ci = cluster_bootstrap_column_means(
            summary,
            cluster_col="query_match_id",
            metrics=["score"],
            n_boot=bootstrap_samples,
            seed=seed + bucket_seed % 10000,
        )
        ci_rows.append({
            "temporal_bucket": str(bucket),
            "score_ci_low": ci.get("score_ci_low", float("nan")),
            "score_ci_high": ci.get("score_ci_high", float("nan")),
        })
    return out.merge(pd.DataFrame(ci_rows), on="temporal_bucket", how="left")


def evaluate_temporal_alignment(
    predictions: pd.DataFrame,
    *,
    tick_rate: float,
    bucket_edges_s: list[float],
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    pred = add_temporal_offsets(predictions, tick_rate=tick_rate)
    if "query_match_id" not in pred.columns:
        pred["query_match_id"] = pred.get("match_id", pred["query_eval_window_id"]).astype(str).str.split("__").str[0]
    pred = bucket_temporal_offsets(pred, edges=bucket_edges_s)
    labels = pred["label"].to_numpy(dtype=np.int32)
    scores = pred["score"].to_numpy(dtype=np.float64)
    q = query_summary(pred)
    metrics = {
        "rows": int(len(pred)),
        "queries": int(pred["candidate_set_id"].nunique()),
        "tick_rate": float(tick_rate),
        "auc": binary_auc(labels, scores),
        "ap": binary_average_precision(labels, scores),
        "positive_score_mean": float(pred.loc[pred["label"].astype(int) == 1, "score"].mean()),
        "negative_score_mean": float(pred.loc[pred["label"].astype(int) == 0, "score"].mean()),
        "negative_offset_s_mean": float(pred.loc[pred["label"].astype(int) == 0, "temporal_offset_s"].mean()),
        "git_commit": git_commit(),
    }
    if not q.empty:
        metrics.update({
            "hit@1": float(q["hit@1"].mean()),
            "hit@5": float(q["hit@5"].mean()),
            "positive_minus_top_negative_mean": float(q["positive_minus_top_negative"].mean()),
            "top_negative_offset_s_mean": float(q["top_negative_offset_s"].mean()),
        })
        if bootstrap_samples > 0:
            metrics.update(cluster_bootstrap_column_means(
                q,
                cluster_col="query_match_id",
                metrics=["hit@1", "hit@5", "positive_score", "top_negative_score", "positive_minus_top_negative"],
                n_boot=bootstrap_samples,
                seed=seed,
            ))
    buckets = bucket_summary(pred, bootstrap_samples=bootstrap_samples, seed=seed + 17)
    metrics["predictions_sha256"] = dataframe_sha256(pred)
    metrics["bucket_rows"] = int(len(buckets))
    return metrics, pred, buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tick-rate", type=float, default=64.0)
    parser.add_argument(
        "--bucket-edges-s",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 8.0, 16.0, 32.0, 1e9],
        help="Temporal offset bucket edges in seconds for negative candidates.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if sorted(args.bucket_edges_s) != list(args.bucket_edges_s) or args.bucket_edges_s[0] != 0:
        raise ValueError("--bucket-edges-s must be sorted and start at 0")
    metrics, pred, buckets = evaluate_temporal_alignment(
        read_parquet(args.predictions),
        tick_rate=args.tick_rate,
        bucket_edges_s=[float(x) for x in args.bucket_edges_s],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    pred_path = args.out / "temporal_alignment_predictions.parquet"
    buckets_path = args.out / "temporal_alignment_buckets.parquet"
    metrics_path = args.out / "metrics_temporal_alignment.json"
    pred.to_parquet(pred_path, index=False)
    buckets.to_parquet(buckets_path, index=False)
    metrics.update({
        "bucket_edges_s": [float(x) for x in args.bucket_edges_s],
        "bucket_table_sha256": dataframe_sha256(buckets),
    })
    write_json(metrics_path, metrics)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
