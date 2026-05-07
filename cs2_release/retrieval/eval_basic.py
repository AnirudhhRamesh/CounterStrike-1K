"""Evaluate cross-POV video retrieval from frozen embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.embeddings import load_embedding_table
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import finite_row_mask, retrieval_metrics
from cs2_release.core.stats import (
    cluster_bootstrap_column_means,
    cluster_bootstrap_mean_delta,
    ensure_query_match_id,
    hypergeom_hit_probability,
    one_positive_chance_topk,
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
    metrics = retrieval_metrics(_prediction_groups(pred_df))
    groups = _prediction_groups(pred_df)
    if groups:
        metrics["chance_top1"] = float(np.mean([one_positive_chance_topk(len(scores), 1) for scores, _ in groups]))
        metrics["chance_top5"] = float(np.mean([one_positive_chance_topk(len(scores), 5) for scores, _ in groups]))
    return metrics


def query_hit_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_set_id, group in pred_df.groupby("candidate_set_id", sort=False):
        n = int(len(group))
        p = int(group["label"].sum())
        positive_ranks = group.loc[group["label"] == 1, "rank"].to_numpy(dtype=np.float64)
        rank = float(np.min(positive_ranks)) if len(positive_ranks) else float("nan")
        rows.append({
            "candidate_set_id": candidate_set_id,
            "query_match_id": str(group["query_match_id"].iloc[0]),
            "top1": float(rank <= 1) if np.isfinite(rank) else float("nan"),
            "top5": float(rank <= 5) if np.isfinite(rank) else float("nan"),
            "mrr": float(1.0 / rank) if np.isfinite(rank) and rank > 0 else float("nan"),
            "chance_top1": hypergeom_hit_probability(n, p, 1),
            "chance_top5": hypergeom_hit_probability(n, p, 5),
        })
    return pd.DataFrame(rows)


def _single_value(df: pd.DataFrame, column: str, default: str = "") -> str:
    if column not in df.columns or df.empty:
        return default
    values = [str(value) for value in df[column].dropna().unique().tolist()]
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    return "mixed:" + ",".join(sorted(values)[:8])


def retrieval_task_metadata(pred_df: pd.DataFrame, *, multipositive: bool = False) -> dict[str, float | int | str]:
    """Return reviewer-facing task semantics and difficulty diagnostics."""

    if pred_df.empty:
        return {}
    pair_policy = _single_value(pred_df, "pair_policy", "same_time_one_positive_pov")
    negative_policy = _single_value(pred_df, "hard_negative_policy", "same_map_phase_different_round")
    positive_radius = (
        float(pred_df["positive_radius"].dropna().iloc[0])
        if "positive_radius" in pred_df.columns and pred_df["positive_radius"].notna().any()
        else float("nan")
    )
    if pair_policy == "same_time_spatially_colocated_pov":
        positive_definition = (
            "synchronized co-located POV"
            if not np.isfinite(positive_radius)
            else f"synchronized co-located POV within {positive_radius:g} CS2 units"
        )
    elif pair_policy == "same_time_any_other_pov":
        positive_definition = "any synchronized non-query POV"
    else:
        positive_definition = "one synchronized non-query POV"

    negative_descriptions = {
        "same_map_phase_different_round": "same map and round phase, different round",
        "same_match_wrong_round": "same match, wrong round",
        "same_round_wrong_time": "same round, wrong time",
        "same_time_far_location": "same synchronized time, far from the query POV",
        "same_location_wrong_time": "same map phase and nearby location, wrong time",
    }
    group_sizes = pred_df.groupby("candidate_set_id", sort=False).size()
    positives = pred_df.groupby("candidate_set_id", sort=False)["label"].sum()
    out: dict[str, float | int | str] = {
        "task/metric": (
            "rank candidate POVs by frozen video-embedding cosine similarity; "
            "retrieve at least one positive"
        ),
        "task/positive_definition": positive_definition,
        "task/negative_policy": negative_policy,
        "task/negative_definition": negative_descriptions.get(negative_policy, negative_policy),
        "task/pair_policy": pair_policy,
        "diagnostic/candidate_sets": int(len(group_sizes)),
        "diagnostic/mean_candidates": float(group_sizes.mean()),
        "diagnostic/mean_positives": float(positives.mean()),
        "diagnostic/chance_hit@1": float(np.mean([
            hypergeom_hit_probability(int(n), int(p), 1)
            for n, p in zip(group_sizes.to_numpy(), positives.to_numpy(), strict=True)
        ])),
        "diagnostic/chance_hit@5": float(np.mean([
            hypergeom_hit_probability(int(n), int(p), 5)
            for n, p in zip(group_sizes.to_numpy(), positives.to_numpy(), strict=True)
        ])),
        "diagnostic/min_candidates": int(group_sizes.min()),
        "diagnostic/max_candidates": int(group_sizes.max()),
        "diagnostic/same_match_negatives": int(negative_policy in {"same_match_wrong_round", "same_round_wrong_time"}),
        "diagnostic/same_round_negatives": int(negative_policy == "same_round_wrong_time"),
        "diagnostic/same_time_negatives": int(negative_policy == "same_time_far_location"),
        "diagnostic/same_location_negatives": int(negative_policy == "same_location_wrong_time"),
        "diagnostic/map_phase_constrained_negatives": int(negative_policy == "same_map_phase_different_round"),
        "diagnostic/multipositive": int(multipositive),
    }
    if "map_slug" in pred_df.columns:
        out["diagnostic/maps"] = int(pred_df["map_slug"].nunique())
    if "query_match_id" in pred_df.columns:
        out["diagnostic/query_matches"] = int(pred_df["query_match_id"].astype(str).nunique())
    return out


def evaluate_retrieval(
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
    metrics = retrieval_metrics(groups)
    pred_df = pd.DataFrame(predictions)
    if not pred_df.empty:
        metrics.update(metrics_from_predictions(pred_df))
        for map_slug, map_group in pred_df.groupby("map_slug", sort=True):
            for key, value in metrics_from_predictions(map_group).items():
                metrics[f"map/{map_slug}/{key}"] = value
        if "hard_negative_policy" in pred_df.columns:
            for policy, policy_group in pred_df.groupby("hard_negative_policy", sort=True):
                for key, value in metrics_from_predictions(policy_group).items():
                    metrics[f"policy/{policy}/{key}"] = value
    return metrics, pred_df


def attach_embedding_ids(pairs: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    row_map = index[["eval_window_id", "sample_key", "pov_idx", "embedding_row_id"]].copy()
    left = row_map.rename(columns={
        "eval_window_id": "query_eval_window_id",
        "sample_key": "query_sample_key",
        "pov_idx": "query_pov_idx",
        "embedding_row_id": "query_embedding_row_id",
    })
    right = row_map.rename(columns={
        "eval_window_id": "candidate_eval_window_id",
        "sample_key": "candidate_sample_key",
        "pov_idx": "candidate_pov_idx",
        "embedding_row_id": "candidate_embedding_row_id",
    })
    out = pairs.merge(
        left,
        on=["query_eval_window_id", "query_pov_idx"],
        how="left",
        suffixes=("", "_query_idx"),
    )
    out = out.merge(
        right,
        on=["candidate_eval_window_id", "candidate_pov_idx"],
        how="left",
        suffixes=("", "_candidate_idx"),
    )
    out = out.dropna(subset=["query_embedding_row_id", "candidate_embedding_row_id"]).copy()
    out["query_embedding_row_id"] = out["query_embedding_row_id"].astype(np.int64)
    out["candidate_embedding_row_id"] = out["candidate_embedding_row_id"].astype(np.int64)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Directory containing embedding_index.parquet and embeddings.npz.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    add_wandb_args(parser)
    args = parser.parse_args()

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(args, job_type="eval-retrieval", config=vars(args))

    pairs = read_parquet(args.pairs)
    index, embeddings = load_embedding_table(args.embeddings)
    pairs = attach_embedding_ids(pairs, index)
    metrics, predictions = evaluate_retrieval(pairs, embeddings)
    if not predictions.empty:
        predictions = ensure_query_match_id(predictions)
        metrics.update(retrieval_task_metadata(predictions, multipositive=False))
    if not predictions.empty and args.bootstrap_samples > 0:
        summary = query_hit_summary(predictions)
        metrics.update(cluster_bootstrap_column_means(
            summary,
            cluster_col="query_match_id",
            metrics=["top1", "top5", "mrr"],
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ))
        metrics.update(cluster_bootstrap_mean_delta(
            summary,
            cluster_col="query_match_id",
            value_col="top1",
            baseline_col="chance_top1",
            prefix="top1_minus_chance",
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed + 17,
        ))
        metrics.update(cluster_bootstrap_mean_delta(
            summary,
            cluster_col="query_match_id",
            value_col="top5",
            baseline_col="chance_top5",
            prefix="top5_minus_chance",
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
            name="cs2-retrieval-eval",
            artifact_type="eval-results",
            paths=[metrics_path, predictions_path],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
