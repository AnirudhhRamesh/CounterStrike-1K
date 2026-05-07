"""Evaluate retrieval candidate sets with one or more positives per query."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.embeddings import load_embedding_table
from cs2_release.retrieval.eval_basic import attach_embedding_ids, retrieval_task_metadata
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import finite_row_mask, multipositive_retrieval_metrics
from cs2_release.core.stats import (
    cluster_bootstrap_column_means,
    cluster_bootstrap_mean_delta,
    ensure_query_match_id,
    hypergeom_hit_probability,
)
from cs2_release.core.tracking import add_wandb_args, finish_wandb, log_artifact, log_dataframe, log_metrics


def _prediction_groups(pred_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    if pred_df.empty:
        return []
    group_cols = ["candidate_set_id"]
    if "__bootstrap_draw" in pred_df.columns:
        group_cols.append("__bootstrap_draw")
    return [
        (
            group["score"].to_numpy(dtype=np.float32),
            group["label"].to_numpy(dtype=np.int32),
        )
        for _, group in pred_df.groupby(group_cols, sort=False)
    ]


def metrics_from_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    return multipositive_retrieval_metrics(_prediction_groups(pred_df))


def query_hit_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_set_id, group in pred_df.groupby("candidate_set_id", sort=False):
        n = int(len(group))
        p = int(group["label"].sum())
        ranked = group.sort_values("rank")["label"].to_numpy(dtype=bool)
        pos = np.flatnonzero(ranked)
        if len(pos):
            first_rank = int(pos[0]) + 1
            hit_count = np.cumsum(ranked)
            ranks = np.arange(1, len(ranked) + 1)
            ap = float((hit_count[ranked] / ranks[ranked]).mean())
            mrr = float(1.0 / first_rank)
        else:
            ap = float("nan")
            mrr = float("nan")
        rows.append({
            "candidate_set_id": candidate_set_id,
            "query_match_id": str(group["query_match_id"].iloc[0]),
            "hit@1": float(((group["label"] == 1) & (group["rank"] <= 1)).any()),
            "hit@5": float(((group["label"] == 1) & (group["rank"] <= 5)).any()),
            "mrr": mrr,
            "map": ap,
            "chance_hit@1": hypergeom_hit_probability(n, p, 1),
            "chance_hit@5": hypergeom_hit_probability(n, p, 5),
        })
    return pd.DataFrame(rows)


def evaluate_multipositive_retrieval(
    pairs: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    finite = finite_row_mask(embeddings)
    predictions = []
    groups = []
    for candidate_set_id, group in pairs.groupby("candidate_set_id", sort=False):
        query_id = int(group["query_embedding_row_id"].iloc[0])
        candidate_ids = group["candidate_embedding_row_id"].to_numpy(dtype=np.int64)
        labels = group["label"].to_numpy(dtype=np.int32)
        if not finite[query_id] or not finite[candidate_ids].all():
            continue
        q = embeddings[query_id].astype(np.float32)
        c = embeddings[candidate_ids].astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        scores = c @ q
        groups.append((scores, labels))
        order = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=np.int32)
        ranks[order] = np.arange(1, len(scores) + 1)
        for row_idx, (_, row) in enumerate(group.iterrows()):
            predictions.append({
                **row.to_dict(),
                "score": float(scores[row_idx]),
                "rank": int(ranks[row_idx]),
            })

    metrics = multipositive_retrieval_metrics(groups)
    pred_df = pd.DataFrame(predictions)
    if not pred_df.empty:
        metrics.update(metrics_from_predictions(pred_df))
        for map_slug, map_group in pred_df.groupby("map_slug", sort=True):
            for key, value in metrics_from_predictions(map_group).items():
                metrics[f"map/{map_slug}/{key}"] = value
        if "pair_policy" in pred_df.columns:
            for policy, policy_group in pred_df.groupby("pair_policy", sort=True):
                for key, value in metrics_from_predictions(policy_group).items():
                    metrics[f"policy/{policy}/{key}"] = value
        if "hard_negative_policy" in pred_df.columns:
            for policy, policy_group in pred_df.groupby("hard_negative_policy", sort=True):
                for key, value in metrics_from_predictions(policy_group).items():
                    metrics[f"negative/{policy}/{key}"] = value
    return metrics, pred_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    add_wandb_args(parser)
    args = parser.parse_args()

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(args, job_type="eval-multipositive-retrieval", config=vars(args))
    pairs = read_parquet(args.pairs)
    index, embeddings = load_embedding_table(args.embeddings)
    pairs = attach_embedding_ids(pairs, index)
    metrics, predictions = evaluate_multipositive_retrieval(pairs, embeddings)
    if not predictions.empty:
        predictions = ensure_query_match_id(predictions)
        metrics.update(retrieval_task_metadata(predictions, multipositive=True))
    if not predictions.empty and args.bootstrap_samples > 0:
        summary = query_hit_summary(predictions)
        metrics.update(cluster_bootstrap_column_means(
            summary,
            cluster_col="query_match_id",
            metrics=["hit@1", "hit@5", "mrr", "map"],
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ))
        metrics.update(cluster_bootstrap_mean_delta(
            summary,
            cluster_col="query_match_id",
            value_col="hit@1",
            baseline_col="chance_hit@1",
            prefix="hit@1_minus_chance",
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed + 17,
        ))
        metrics.update(cluster_bootstrap_mean_delta(
            summary,
            cluster_col="query_match_id",
            value_col="hit@5",
            baseline_col="chance_hit@5",
            prefix="hit@5_minus_chance",
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed + 23,
        ))

    out_dir = args.out if args.out.suffix == "" else args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "retrieval_predictions.parquet"
    metrics_path = out_dir / "metrics_retrieval.json"
    predictions.to_parquet(predictions_path, index=False)
    metrics.update({
        "pairs": int(len(pairs)),
        "predictions": int(len(predictions)),
        "pairs_sha256": dataframe_sha256(pairs),
        "embedding_rows": int(len(index)),
        "embedding_dim": int(embeddings.shape[1]),
        "git_commit": git_commit(),
    })
    write_json(metrics_path, metrics)
    log_metrics(wandb_run, metrics, prefix="retrieval", summary=True)
    if not predictions.empty:
        preview = predictions.sort_values(["candidate_set_id", "rank"]).head(500)
        log_dataframe(wandb_run, "retrieval/top_ranked_predictions", preview, max_rows=500)
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-multipositive-retrieval-eval",
            artifact_type="eval-results",
            paths=[metrics_path, predictions_path],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
