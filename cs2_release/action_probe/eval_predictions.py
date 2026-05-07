"""Add match-clustered confidence intervals to saved action-probe predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.stats import cluster_bootstrap, ensure_query_match_id
from cs2_release.action_probe.train_video import metrics_from_prediction_frame


def _label_cols(predictions: pd.DataFrame, prefix: str) -> list[str]:
    labels = []
    for column in predictions.columns:
        if column.startswith(prefix) and column.endswith("_target"):
            label = column.removesuffix("_target")
            if f"{label}_prob" in predictions.columns:
                labels.append(label)
    return sorted(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="Optional existing metrics_action_probe.json to update under the test key.")
    parser.add_argument("--label-prefix", default="label_")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args()

    predictions = read_parquet(args.predictions)
    label_cols = _label_cols(predictions, args.label_prefix)
    if not label_cols:
        raise ValueError(f"no prediction target/probability columns with prefix {args.label_prefix!r}")
    metrics = metrics_from_prediction_frame(predictions, label_cols)
    predictions = ensure_query_match_id(predictions)
    cluster_col = "match_id" if "match_id" in predictions.columns else "query_match_id"
    if args.bootstrap_samples > 0 and cluster_col in predictions.columns:
        metric_keys = ["macro_auc", "macro_ap"]
        for label in label_cols:
            metric_keys.extend([f"{label}/auc", f"{label}/ap"])
        metrics.update(cluster_bootstrap(
            predictions,
            cluster_col=cluster_col,
            metric_fn=lambda df: metrics_from_prediction_frame(df, label_cols),
            metrics=metric_keys,
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ))

    if args.metrics_json is not None and args.metrics_json.exists():
        payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
        payload["test"] = {**payload.get("test", {}), **metrics}
    else:
        payload = {"test": metrics}
    payload.update({
        "prediction_rows": int(len(predictions)),
        "prediction_sha256": dataframe_sha256(predictions),
        "label_cols": label_cols,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "git_commit": git_commit(),
    })
    write_json(args.out, payload)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
