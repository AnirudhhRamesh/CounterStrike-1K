"""Evaluate whether frozen video embeddings separate co-located from far POVs.

This is a dataset/evaluation diagnostic, not a trained model baseline. Given
spatial retrieval pairs, it scores each query-candidate pair with cosine
similarity between frozen video embeddings and reports binary AUROC/AP for
near/co-located positives versus the configured hard negatives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.embeddings import load_embedding_table
from cs2_release.retrieval.eval_basic import attach_embedding_ids, retrieval_task_metadata
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import binary_auc, binary_average_precision, finite_row_mask
from cs2_release.core.stats import cluster_bootstrap, ensure_query_match_id
from cs2_release.core.tracking import add_wandb_args, finish_wandb, log_artifact, log_dataframe, log_metrics


def metrics_from_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {"auc": float("nan"), "ap": float("nan"), "prevalence": float("nan")}
    labels = predictions["label"].to_numpy(dtype=np.float32)
    scores = predictions["score"].to_numpy(dtype=np.float32)
    return {
        "auc": binary_auc(labels, scores),
        "ap": binary_average_precision(labels, scores),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
    }


def evaluate_pairwise_spatial(pairs: pd.DataFrame, embeddings: np.ndarray) -> tuple[dict, pd.DataFrame]:
    finite = finite_row_mask(embeddings)
    query_ids = pairs["query_embedding_row_id"].to_numpy(dtype=np.int64)
    candidate_ids = pairs["candidate_embedding_row_id"].to_numpy(dtype=np.int64)
    keep = finite[query_ids] & finite[candidate_ids]
    predictions = pairs.loc[keep].copy().reset_index(drop=True)
    query_ids = query_ids[keep]
    candidate_ids = candidate_ids[keep]
    scores = np.empty(len(predictions), dtype=np.float32)
    chunk_size = 65536
    for start in range(0, len(predictions), chunk_size):
        end = min(start + chunk_size, len(predictions))
        q = embeddings[query_ids[start:end]].astype(np.float32)
        c = embeddings[candidate_ids[start:end]].astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        scores[start:end] = np.sum(q * c, axis=1)
    predictions["score"] = scores.astype(float)
    metrics = metrics_from_predictions(predictions)
    if not predictions.empty:
        metrics.update(retrieval_task_metadata(predictions, multipositive=True))
        if "hard_negative_policy" in predictions.columns:
            for policy, policy_group in predictions.groupby("hard_negative_policy", sort=True):
                for key, value in metrics_from_predictions(policy_group).items():
                    metrics[f"negative/{policy}/{key}"] = value
        if "query_candidate_distance" in predictions.columns:
            pos_dist = predictions.loc[predictions["label"] == 1, "query_candidate_distance"]
            neg_dist = predictions.loc[predictions["label"] == 0, "query_candidate_distance"]
            metrics["distance/positive_mean"] = float(pos_dist.mean()) if len(pos_dist) else float("nan")
            metrics["distance/negative_mean"] = float(neg_dist.mean()) if len(neg_dist) else float("nan")
    return metrics, predictions


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

    wandb_run = init_wandb(args, job_type="eval-pairwise-spatial-probe", config=vars(args))
    pairs = read_parquet(args.pairs)
    index, embeddings = load_embedding_table(args.embeddings)
    pairs = attach_embedding_ids(pairs, index)
    metrics, predictions = evaluate_pairwise_spatial(pairs, embeddings)
    if not predictions.empty:
        predictions = ensure_query_match_id(predictions)
    if not predictions.empty and args.bootstrap_samples > 0:
        metrics.update(cluster_bootstrap(
            predictions,
            cluster_col="query_match_id",
            metric_fn=metrics_from_predictions,
            metrics=["auc", "ap", "prevalence"],
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ))

    out_dir = args.out if args.out.suffix == "" else args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "pairwise_spatial_predictions.parquet"
    metrics_path = out_dir / "metrics_pairwise_spatial_probe.json"
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
    log_metrics(wandb_run, metrics, prefix="pairwise_spatial", summary=True)
    if not predictions.empty:
        preview = predictions.sort_values(["candidate_set_id", "score"], ascending=[True, False]).head(500)
        log_dataframe(wandb_run, "pairwise_spatial/predictions", preview, max_rows=500)
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-pairwise-spatial-probe",
            artifact_type="eval-results",
            paths=[metrics_path, predictions_path],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
