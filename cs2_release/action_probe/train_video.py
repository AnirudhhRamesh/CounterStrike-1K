"""Train a lightweight multi-label action probe on frozen video embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cs2_release.core.embeddings import load_embedding_table
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import binary_auc, binary_average_precision
from cs2_release.core.stats import cluster_bootstrap
from cs2_release.core.tracking import add_wandb_args, finish_wandb, log_artifact, log_dataframe, log_metrics


class ActionProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def attach_labels(labels: pd.DataFrame, embedding_index: pd.DataFrame) -> pd.DataFrame:
    row_map = embedding_index[["eval_window_id", "sample_key", "pov_idx", "embedding_row_id"]].copy()
    return labels.merge(
        row_map,
        on=["eval_window_id", "sample_key", "pov_idx"],
        how="inner",
    )


def metrics_from_prediction_frame(pred_df: pd.DataFrame, label_cols: list[str]) -> dict[str, float]:
    """Compute action-probe metrics from a saved prediction table."""

    metrics: dict[str, float] = {}
    aucs = []
    aps = []
    for label in label_cols:
        target_col = f"{label}_target"
        prob_col = f"{label}_prob"
        if target_col not in pred_df.columns or prob_col not in pred_df.columns:
            continue
        y = pred_df[target_col].to_numpy(dtype=np.float32)
        probs = pred_df[prob_col].to_numpy(dtype=np.float32)
        auc = binary_auc(y, probs)
        ap = binary_average_precision(y, probs)
        metrics[f"{label}/auc"] = auc
        metrics[f"{label}/ap"] = ap
        metrics[f"{label}/prevalence"] = float(np.mean(y)) if len(y) else float("nan")
        if np.isfinite(auc):
            aucs.append(float(auc))
        if np.isfinite(ap):
            aps.append(float(ap))
    metrics["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    metrics["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    label_cols: list[str],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    if len(x) == 0:
        return {"macro_auc": float("nan"), "macro_ap": float("nan")}, np.empty_like(y)
    logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    metrics: dict[str, float] = {}
    aucs = []
    aps = []
    for idx, label in enumerate(label_cols):
        auc = binary_auc(y[:, idx], probs[:, idx])
        ap = binary_average_precision(y[:, idx], probs[:, idx])
        prevalence = float(y[:, idx].mean()) if len(y) else float("nan")
        metrics[f"{label}/auc"] = auc
        metrics[f"{label}/ap"] = ap
        metrics[f"{label}/prevalence"] = prevalence
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(ap):
            aps.append(ap)
    metrics["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    metrics["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return metrics, probs


def _split_arrays(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_cols: list[str],
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split_df = df[df["split"].astype(str) == split].copy()
    if split_df.empty:
        return (
            np.empty((0, embeddings.shape[1]), dtype=np.float32),
            np.empty((0, len(label_cols)), dtype=np.float32),
            split_df,
        )
    row_ids = split_df["embedding_row_id"].to_numpy(dtype=np.int64)
    x = embeddings[row_ids].astype(np.float32)
    y = split_df[label_cols].to_numpy(dtype=np.float32)
    finite = np.isfinite(x).all(axis=1)
    return x[finite], y[finite], split_df.loc[finite].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-prefix", default="label_")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    add_wandb_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    labels_raw = read_parquet(args.labels)
    label_cols = [c for c in labels_raw.columns if c.startswith(args.label_prefix)]
    if not label_cols:
        raise ValueError(f"no columns with prefix {args.label_prefix!r}")
    index, embeddings = load_embedding_table(args.embeddings)
    labels = attach_labels(labels_raw, index)
    if labels.empty:
        raise RuntimeError("no labels matched embedding rows")

    x_train, y_train, train_df = _split_arrays(labels, embeddings, label_cols, "train")
    x_val, y_val, _ = _split_arrays(labels, embeddings, label_cols, "val")
    x_test, y_test, test_df = _split_arrays(labels, embeddings, label_cols, "test")
    if len(x_train) == 0:
        raise RuntimeError("no train labels matched embedding rows")

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(
        args,
        job_type="train-action-probe",
        config={
            **vars(args),
            "device_resolved": str(device),
            "embedding_rows": int(len(index)),
            "embedding_dim": int(embeddings.shape[1]),
            "label_cols": label_cols,
            "train_examples": int(len(x_train)),
            "val_examples": int(len(x_val)),
            "test_examples": int(len(x_test)),
        },
    )

    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = np.divide(neg, np.maximum(pos, 1.0)).astype(np.float32)
    model = ActionProbeMLP(
        input_dim=int(embeddings.shape[1]),
        output_dim=len(label_cols),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_score = -1.0
    best_state = None
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        started = time.time()
        for x, y in loader:
            global_step += 1
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        train_metrics, _ = evaluate(model, x_train, y_train, label_cols, device)
        val_metrics, _ = evaluate(model, x_val, y_val, label_cols, device)
        monitor = val_metrics.get("macro_auc")
        if not np.isfinite(monitor):
            monitor = train_metrics.get("macro_auc", float("nan"))
        if np.isfinite(monitor) and float(monitor) > best_score:
            best_score = float(monitor)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "epoch_s": float(time.time() - started),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        log_metrics(wandb_run, record, prefix="epoch", step=global_step)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_metrics, _ = evaluate(model, x_train, y_train, label_cols, device)
    val_metrics, _ = evaluate(model, x_val, y_val, label_cols, device)
    test_metrics, test_probs = evaluate(model, x_test, y_test, label_cols, device)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "action_probe.pt"
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": int(embeddings.shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "label_cols": label_cols,
        "seed": args.seed,
    }, ckpt_path)
    pred_cols = [
        "eval_window_id", "sample_key", "pov_idx", "split", "map_slug",
        "match_id", "round_id", "start_tick", "end_tick",
    ]
    pred_df = test_df[[c for c in pred_cols if c in test_df.columns]].copy()
    for idx, label in enumerate(label_cols):
        pred_df[f"{label}_target"] = y_test[:, idx] if len(y_test) else []
        pred_df[f"{label}_prob"] = test_probs[:, idx] if len(test_probs) else []
    if not pred_df.empty and args.bootstrap_samples > 0 and "match_id" in pred_df.columns:
        metric_keys = ["macro_auc", "macro_ap"]
        for label in label_cols:
            metric_keys.extend([f"{label}/auc", f"{label}/ap"])
        test_metrics.update(cluster_bootstrap(
            pred_df,
            cluster_col="match_id",
            metric_fn=lambda df: metrics_from_prediction_frame(df, label_cols),
            metrics=metric_keys,
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ))
    predictions_path = args.out / "action_probe_predictions.parquet"
    pred_df.to_parquet(predictions_path, index=False)
    metrics = {
        "checkpoint": str(ckpt_path),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "label_cols": label_cols,
        "train_examples": int(len(x_train)),
        "val_examples": int(len(x_val)),
        "test_examples": int(len(x_test)),
        "labels_sha256": dataframe_sha256(labels_raw),
        "git_commit": git_commit(),
    }
    metrics_path = args.out / "metrics_action_probe.json"
    write_json(metrics_path, metrics)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    log_metrics(wandb_run, metrics, prefix="action_probe", summary=True)
    if not pred_df.empty:
        log_dataframe(wandb_run, "action_probe/test_predictions", pred_df.head(500), max_rows=500)
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-action-probe",
            artifact_type="model",
            paths=[ckpt_path, metrics_path, predictions_path, args.out / "config.json"],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
