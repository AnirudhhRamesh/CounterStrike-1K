"""Train k-POV probes for actions occurring only in unobserved synchronized POVs."""

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
from cs2_release.action_probe.train_multipov import MultiPovProbe, aggregate_features
from cs2_release.core.tracking import add_wandb_args, finish_wandb, log_artifact, log_dataframe, log_metrics


def attach_embedding_rows(labels: pd.DataFrame, embedding_index: pd.DataFrame) -> pd.DataFrame:
    row_map = embedding_index[["eval_window_id", "sample_key", "pov_idx", "embedding_row_id"]].copy()
    out = labels.merge(row_map, on=["eval_window_id", "sample_key", "pov_idx"], how="inner")
    if out.empty:
        raise RuntimeError("no action labels matched embedding rows")
    return out


def build_examples(
    labels: pd.DataFrame,
    embeddings: np.ndarray,
    label_cols: list[str],
    *,
    split: str,
    k_povs: int,
    subsets_per_window: int,
    seed: int,
    exclude_observed_positive: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    features = []
    targets = []
    masks = []
    rows = []
    split_df = labels[labels["split"].astype(str) == split].copy()
    for eval_window_id, group in split_df.groupby("eval_window_id", sort=False):
        group = group.sort_values("pov_idx").reset_index(drop=True)
        row_ids = group["embedding_row_id"].to_numpy(dtype=np.int64)
        finite = np.isfinite(embeddings[row_ids]).all(axis=1)
        group = group.loc[finite].reset_index(drop=True)
        row_ids = group["embedding_row_id"].to_numpy(dtype=np.int64)
        if len(row_ids) < 2:
            continue
        kk = min(int(k_povs), len(row_ids) - 1)
        if kk <= 0:
            continue
        label_values = group[label_cols].to_numpy(dtype=np.float32)
        for subset_idx in range(int(subsets_per_window)):
            if subset_idx == 0:
                chosen = np.arange(kk, dtype=np.int64)
            else:
                chosen = np.sort(rng.choice(np.arange(len(row_ids)), size=kk, replace=False))
            unselected_mask = np.ones(len(row_ids), dtype=bool)
            unselected_mask[chosen] = False
            selected_labels = label_values[chosen].max(axis=0)
            unseen_labels = label_values[unselected_mask].max(axis=0)
            metric_mask = np.ones(len(label_cols), dtype=np.float32)
            if exclude_observed_positive:
                metric_mask = (selected_labels == 0).astype(np.float32)
            if metric_mask.sum() == 0:
                continue
            x = embeddings[row_ids[chosen]].astype(np.float32)
            features.append(aggregate_features(x))
            targets.append(unseen_labels.astype(np.float32))
            masks.append(metric_mask)
            rows.append({
                "eval_window_id": str(eval_window_id),
                "match_id": str(group["match_id"].iloc[0]),
                "round_id": str(group["round_id"].iloc[0]),
                "split": str(split),
                "map_slug": str(group["map_slug"].iloc[0]),
                "k_povs": int(kk),
                "unobserved_povs": int(len(row_ids) - kk),
                "subset_idx": int(subset_idx),
                "selected_povs": ",".join(str(int(v)) for v in group.loc[chosen, "pov_idx"].tolist()),
            })
    if not features:
        return (
            np.empty((0, embeddings.shape[1] * 3), dtype=np.float32),
            np.empty((0, len(label_cols)), dtype=np.float32),
            np.empty((0, len(label_cols)), dtype=np.float32),
            pd.DataFrame(rows),
        )
    return np.stack(features), np.stack(targets), np.stack(masks), pd.DataFrame(rows)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 0), dtype=np.float32)
    logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def masked_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray, label_cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    aucs = []
    aps = []
    for idx, label in enumerate(label_cols):
        valid = mask[:, idx] > 0.5
        yy = y[valid, idx]
        pp = p[valid, idx]
        auc = binary_auc(yy, pp)
        ap = binary_average_precision(yy, pp)
        out[f"{label}/auc"] = auc
        out[f"{label}/ap"] = ap
        out[f"{label}/prevalence"] = float(yy.mean()) if len(yy) else float("nan")
        out[f"{label}/examples"] = int(len(yy))
        out[f"{label}/positives"] = int(yy.sum()) if len(yy) else 0
        if np.isfinite(auc):
            aucs.append(float(auc))
        if np.isfinite(ap):
            aps.append(float(ap))
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    out["masked_label_examples"] = int(mask.sum())
    return out


def metrics_from_predictions(pred_df: pd.DataFrame, label_cols: list[str]) -> dict[str, float]:
    if pred_df.empty:
        return {"macro_auc": float("nan"), "macro_ap": float("nan")}
    y = np.stack([pred_df[f"{label}_target"].to_numpy(dtype=np.float32) for label in label_cols], axis=1)
    p = np.stack([pred_df[f"{label}_prob"].to_numpy(dtype=np.float32) for label in label_cols], axis=1)
    m = np.stack([pred_df[f"{label}_mask"].to_numpy(dtype=np.float32) for label in label_cols], axis=1)
    return masked_metrics(y, p, m, label_cols)


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    denom = torch.clamp(mask.sum(), min=1.0)
    return (loss * mask).sum() / denom


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, [(x - mean) / std if len(x) else x for x in others], mean, std


def _pos_weight(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pos = (y * mask).sum(axis=0)
    total = mask.sum(axis=0)
    neg = total - pos
    weights = np.divide(neg, np.maximum(pos, 1.0))
    weights[~np.isfinite(weights)] = 1.0
    return np.maximum(weights, 1.0).astype(np.float32)


def train_one_k(
    *,
    labels: pd.DataFrame,
    embeddings: np.ndarray,
    label_cols: list[str],
    k_povs: int,
    args,
    device: torch.device,
    wandb_run=None,
    step_offset: int = 0,
) -> tuple[dict, pd.DataFrame, dict]:
    x_train, y_train, m_train, _ = build_examples(
        labels, embeddings, label_cols,
        split="train", k_povs=k_povs,
        subsets_per_window=args.train_subsets_per_window,
        seed=args.seed + k_povs,
        exclude_observed_positive=args.exclude_observed_positive,
    )
    x_val, y_val, m_val, _ = build_examples(
        labels, embeddings, label_cols,
        split="val", k_povs=k_povs,
        subsets_per_window=args.eval_subsets_per_window,
        seed=args.seed + 1000 + k_povs,
        exclude_observed_positive=args.exclude_observed_positive,
    )
    x_test, y_test, m_test, test_df = build_examples(
        labels, embeddings, label_cols,
        split="test", k_povs=k_povs,
        subsets_per_window=args.eval_subsets_per_window,
        seed=args.seed + 2000 + k_povs,
        exclude_observed_positive=args.exclude_observed_positive,
    )
    if len(x_train) == 0:
        raise RuntimeError(f"no train examples for k={k_povs}")
    x_train, (x_val, x_test), mean, std = _standardize(x_train, x_val, x_test)
    model = MultiPovProbe(x_train.shape[1], len(label_cols), args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.from_numpy(_pos_weight(y_train, m_train)).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(m_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = masked_bce_loss(model(x), y, mask, pos_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        val_probs = predict(model, x_val, device)
        val_metrics = masked_metrics(y_val, val_probs, m_val, label_cols) if len(x_val) else {}
        score = float(val_metrics.get("macro_auc", float("nan")))
        if np.isfinite(score) and score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        record = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "epoch_s": float(time.time() - started),
            "val_macro_auc": score,
        }
        history.append(record)
        log_metrics(wandb_run, record, prefix=f"k{k_povs}/epoch", step=step_offset + epoch)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_probs = predict(model, x_train, device)
    val_probs = predict(model, x_val, device)
    test_probs = predict(model, x_test, device)
    metrics = {
        "k_povs": int(k_povs),
        "train_examples": int(len(x_train)),
        "val_examples": int(len(x_val)),
        "test_examples": int(len(x_test)),
        "train": masked_metrics(y_train, train_probs, m_train, label_cols),
        "val": masked_metrics(y_val, val_probs, m_val, label_cols) if len(x_val) else {},
        "test": masked_metrics(y_test, test_probs, m_test, label_cols) if len(x_test) else {},
        "history": history,
    }
    pred_df = test_df.copy()
    for idx, label in enumerate(label_cols):
        pred_df[f"{label}_target"] = y_test[:, idx] if len(y_test) else []
        pred_df[f"{label}_mask"] = m_test[:, idx] if len(m_test) else []
        pred_df[f"{label}_prob"] = test_probs[:, idx] if len(test_probs) else []
    if len(pred_df) and args.bootstrap_samples > 0:
        metric_keys = ["macro_auc", "macro_ap"]
        for label in label_cols:
            metric_keys.extend([f"{label}/auc", f"{label}/ap"])
        ci = cluster_bootstrap(
            pred_df,
            cluster_col="match_id",
            metric_fn=lambda df: metrics_from_predictions(df, label_cols),
            metrics=metric_keys,
            n_boot=args.bootstrap_samples,
            seed=args.seed + 3000 + k_povs,
        )
        metrics["test"].update(ci)
    state = {
        "model_state": model.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "label_cols": label_cols,
        "k_povs": int(k_povs),
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
    }
    return metrics, pred_df, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-prefix", default="label_")
    parser.add_argument("--actions", nargs="+", default=["FIRE", "RIGHTCLICK", "RELOAD", "JUMP", "MOUSE_MOVE"])
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--train-subsets-per-window", type=int, default=4)
    parser.add_argument("--eval-subsets-per-window", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude-observed-positive", action=argparse.BooleanOptionalAction, default=True)
    add_wandb_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    labels_raw = read_parquet(args.labels)
    label_cols = [f"{args.label_prefix}{action}" for action in args.actions]
    label_cols = [c for c in label_cols if c in labels_raw.columns]
    if not label_cols:
        raise ValueError("none of the requested action labels are present")
    index, embeddings = load_embedding_table(args.embeddings)
    labels = attach_embedding_rows(labels_raw, index)

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(args, job_type="train-offscreen-action-probe", config=vars(args))
    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    pred_paths = []
    ckpt_paths = []
    for k_order, k_povs in enumerate(args.k_values):
        metrics, pred_df, state = train_one_k(
            labels=labels,
            embeddings=embeddings,
            label_cols=label_cols,
            k_povs=int(k_povs),
            args=args,
            device=device,
            wandb_run=wandb_run,
            step_offset=k_order * (args.epochs + 10),
        )
        all_metrics.append(metrics)
        pred_path = args.out / f"offscreen_action_probe_k{k_povs}_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        pred_paths.append(pred_path)
        ckpt_path = args.out / f"offscreen_action_probe_k{k_povs}.pt"
        torch.save(state, ckpt_path)
        ckpt_paths.append(ckpt_path)
        log_metrics(wandb_run, metrics, prefix=f"k{k_povs}", summary=True)
        if not pred_df.empty:
            log_dataframe(wandb_run, f"k{k_povs}/test_predictions", pred_df.head(500), max_rows=500)

    metrics = {
        "label_cols": label_cols,
        "k_values": [int(k) for k in args.k_values],
        "exclude_observed_positive": bool(args.exclude_observed_positive),
        "embedding_rows": int(len(index)),
        "embedding_dim": int(embeddings.shape[1]),
        "labels_sha256": dataframe_sha256(labels_raw),
        "attached_label_rows": int(len(labels)),
        "results": all_metrics,
        "git_commit": git_commit(),
    }
    metrics_path = args.out / "metrics_offscreen_action_probe.json"
    write_json(metrics_path, metrics)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-offscreen-action-probe",
            artifact_type="model",
            paths=[metrics_path, args.out / "config.json", *pred_paths, *ckpt_paths],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
