"""Train a small log-mel CNN audio probe for action labels.

This is the audio analogue of :mod:`cs2_release.action_probe.train_video`. It reads
log-mel spectrograms produced by :mod:`cs2_release.encoders.extract_audio` plus
multi-label action targets from :mod:`cs2_release.action_probe.labels`,
trains a small Conv2d -> GAP -> MLP head end-to-end, and reports macro AUROC
/ AP with cluster-bootstrapped CIs over ``match_id``.
"""

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

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import binary_auc, binary_average_precision
from cs2_release.core.stats import cluster_bootstrap
from cs2_release.core.tracking import (
    add_wandb_args,
    finish_wandb,
    log_artifact,
    log_dataframe,
    log_metrics,
)


class AudioProbeCNN(nn.Module):
    """Tiny Conv2d encoder over (n_mels, n_time) log-mel inputs."""

    def __init__(
        self,
        *,
        n_mels: int,
        n_time: int,
        output_dim: int,
        channels: tuple[int, int, int] = (32, 64, 128),
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.input_shape = (1, int(n_mels), int(n_time))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        return self.head(z)


def load_audio_features(audio_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    audio_dir = Path(audio_dir)
    index_path = audio_dir / "embedding_index.parquet"
    npz_path = audio_dir / "audio_features.npz"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    index = read_parquet(index_path).reset_index(drop=True)
    payload = np.load(npz_path, allow_pickle=False)
    features = payload["features"].astype(np.float32)
    if len(index) != features.shape[0]:
        raise ValueError(
            f"audio row mismatch: index has {len(index)}, features has {features.shape[0]}"
        )
    index = index.copy()
    index["embedding_row_id"] = np.arange(len(index), dtype=np.int64)
    return index, features


def attach_labels(labels: pd.DataFrame, embedding_index: pd.DataFrame) -> pd.DataFrame:
    row_map = embedding_index[["eval_window_id", "sample_key", "pov_idx", "embedding_row_id"]].copy()
    return labels.merge(
        row_map,
        on=["eval_window_id", "sample_key", "pov_idx"],
        how="inner",
    )


def metrics_from_prediction_frame(pred_df: pd.DataFrame, label_cols: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    aucs: list[float] = []
    aps: list[float] = []
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
def predict(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0, 0), dtype=np.float32)
    out: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).to(device)
        x = (x - feature_mean) / feature_std
        logits = model(x).detach().cpu().numpy()
        out.append(logits)
    return np.concatenate(out, axis=0)


def evaluate(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    label_cols: list[str],
    *,
    batch_size: int,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> tuple[dict[str, float], np.ndarray]:
    if len(features) == 0:
        return ({"macro_auc": float("nan"), "macro_ap": float("nan")},
                np.empty_like(targets, dtype=np.float32))
    logits = predict(
        model,
        features,
        batch_size=batch_size,
        device=device,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    metrics: dict[str, float] = {}
    aucs: list[float] = []
    aps: list[float] = []
    for idx, label in enumerate(label_cols):
        auc = binary_auc(targets[:, idx], probs[:, idx])
        ap = binary_average_precision(targets[:, idx], probs[:, idx])
        prevalence = float(targets[:, idx].mean()) if len(targets) else float("nan")
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
    features: np.ndarray,
    label_cols: list[str],
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split_df = df[df["split"].astype(str) == split].reset_index(drop=True)
    if split_df.empty:
        empty_feats = np.empty((0, *features.shape[1:]), dtype=np.float32)
        empty_targets = np.empty((0, len(label_cols)), dtype=np.float32)
        return empty_feats, empty_targets, split_df
    row_ids = split_df["embedding_row_id"].to_numpy(dtype=np.int64)
    x = features[row_ids].astype(np.float32, copy=False)
    y = split_df[label_cols].to_numpy(dtype=np.float32)
    finite = np.isfinite(x.reshape(x.shape[0], -1)).all(axis=1)
    return x[finite], y[finite], split_df.loc[finite].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True,
                        help="Directory with audio_features.npz + embedding_index.parquet.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-prefix", default="label_")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
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
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    labels_raw = read_parquet(args.labels)
    label_cols = [c for c in labels_raw.columns if c.startswith(args.label_prefix)]
    if not label_cols:
        raise ValueError(f"no label columns with prefix {args.label_prefix!r}")
    index, features = load_audio_features(args.features)
    if features.ndim != 3:
        raise ValueError(
            f"expected (N, n_mels, n_time) audio features; got shape {features.shape}"
        )
    n_mels, n_time = int(features.shape[1]), int(features.shape[2])
    labels = attach_labels(labels_raw, index)
    if labels.empty:
        raise RuntimeError("no labels matched audio rows")

    x_train, y_train, train_df = _split_arrays(labels, features, label_cols, "train")
    x_val, y_val, _val_df = _split_arrays(labels, features, label_cols, "val")
    x_test, y_test, test_df = _split_arrays(labels, features, label_cols, "test")
    if len(x_train) == 0:
        raise RuntimeError("no train labels matched audio rows")

    feature_mean = torch.tensor(float(x_train.mean()), device=device)
    feature_std = torch.tensor(float(x_train.std()) + 1e-6, device=device)

    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = np.divide(neg, np.maximum(pos, 1.0)).astype(np.float32)

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(
        args,
        job_type="train-audio-probe",
        config={
            **vars(args),
            "device_resolved": str(device),
            "feature_shape": [n_mels, n_time],
            "label_cols": label_cols,
            "train_examples": int(len(x_train)),
            "val_examples": int(len(x_val)),
            "test_examples": int(len(x_test)),
            "train_feature_mean": float(feature_mean.item()),
            "train_feature_std": float(feature_std.item()),
        },
    )

    model = AudioProbeCNN(
        n_mels=n_mels,
        n_time=n_time,
        output_dim=len(label_cols),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))

    train_x_tensor = torch.from_numpy(x_train).float()
    train_y_tensor = torch.from_numpy(y_train).float()
    loader = DataLoader(
        TensorDataset(train_x_tensor, train_y_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history: list[dict] = []
    best_score = -1.0
    best_state = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        started = time.time()
        for x, y in loader:
            global_step += 1
            x = x.to(device)
            x = (x - feature_mean) / feature_std
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        train_metrics, _ = evaluate(
            model, x_train, y_train, label_cols,
            batch_size=args.batch_size, device=device,
            feature_mean=feature_mean, feature_std=feature_std,
        )
        val_metrics, _ = evaluate(
            model, x_val, y_val, label_cols,
            batch_size=args.batch_size, device=device,
            feature_mean=feature_mean, feature_std=feature_std,
        )
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
    train_metrics, _ = evaluate(
        model, x_train, y_train, label_cols,
        batch_size=args.batch_size, device=device,
        feature_mean=feature_mean, feature_std=feature_std,
    )
    val_metrics, _ = evaluate(
        model, x_val, y_val, label_cols,
        batch_size=args.batch_size, device=device,
        feature_mean=feature_mean, feature_std=feature_std,
    )
    test_metrics, test_probs = evaluate(
        model, x_test, y_test, label_cols,
        batch_size=args.batch_size, device=device,
        feature_mean=feature_mean, feature_std=feature_std,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "audio_probe.pt"
    torch.save({
        "model_state": model.state_dict(),
        "n_mels": n_mels,
        "n_time": n_time,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "label_cols": label_cols,
        "feature_mean": float(feature_mean.item()),
        "feature_std": float(feature_std.item()),
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
    predictions_path = args.out / "audio_probe_predictions.parquet"
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
        "feature_shape": [n_mels, n_time],
        "labels_sha256": dataframe_sha256(labels_raw),
        "git_commit": git_commit(),
    }
    metrics_path = args.out / "metrics_audio_probe.json"
    write_json(metrics_path, metrics)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    log_metrics(wandb_run, metrics, prefix="audio_probe", summary=True)
    if not pred_df.empty:
        log_dataframe(wandb_run, "audio_probe/test_predictions", pred_df.head(500), max_rows=500)
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-audio-probe",
            artifact_type="model",
            paths=[ckpt_path, metrics_path, predictions_path, args.out / "config.json"],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
