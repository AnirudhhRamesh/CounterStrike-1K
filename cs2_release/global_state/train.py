"""Train k-POV probes for global round/event labels from frozen video embeddings."""

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
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, read_release_parquet, write_json
from cs2_release.core.metrics import binary_auc, binary_average_precision
from cs2_release.core.stats import cluster_bootstrap
from cs2_release.action_probe.train_multipov import MultiPovProbe, aggregate_features
from cs2_release.core.tracking import add_wandb_args, finish_wandb, log_artifact, log_dataframe, log_metrics


DEFAULT_TARGETS = [
    "target_T_WIN",
    "target_BOMB_PLANTED",
    "target_BOMB_EXPLODED",
    "target_HAS_AWP",
    "target_HIGH_KILL_ROUND",
    "target_PISTOL_ROUND",
]


def attach_embedding_rows(windows: pd.DataFrame, embedding_index: pd.DataFrame) -> pd.DataFrame:
    row_map = embedding_index[["eval_window_id", "sample_key", "pov_idx", "embedding_row_id"]].copy()
    out = windows.merge(row_map, on=["eval_window_id", "sample_key", "pov_idx"], how="inner")
    if out.empty:
        raise RuntimeError("no windows matched embedding rows")
    return out


def build_global_targets(windows: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    base_cols = [
        "eval_window_id", "round_id", "match_id", "round_idx", "window_idx",
        "split", "map_slug", "phase_bucket", "start_tick", "end_tick",
    ]
    base = windows[[c for c in base_cols if c in windows.columns]].drop_duplicates("eval_window_id").copy()
    round_index = read_release_parquet(root, "round_index.parquet")
    keep = [
        "round_id", "round_winner", "bomb_planted", "bomb_exploded",
        "round_kills_total", "has_awp_any_pov", "is_pistol_round",
    ]
    keep = [c for c in keep if c in round_index.columns]
    base = base.merge(round_index[keep].drop_duplicates("round_id"), on="round_id", how="left")
    if "round_winner" in base.columns:
        base["target_T_WIN"] = (base["round_winner"].astype(str) == "T").astype(np.int8)
    if "bomb_planted" in base.columns:
        base["target_BOMB_PLANTED"] = base["bomb_planted"].fillna(False).astype(bool).astype(np.int8)
    elif "bomb_planted_round" in windows.columns:
        planted = windows.groupby("eval_window_id")["bomb_planted_round"].max().astype(bool).astype(np.int8)
        base = base.merge(planted.rename("target_BOMB_PLANTED"), on="eval_window_id", how="left")
    if "bomb_exploded" in base.columns:
        base["target_BOMB_EXPLODED"] = base["bomb_exploded"].fillna(False).astype(bool).astype(np.int8)
    elif "bomb_exploded_round" in windows.columns:
        exploded = windows.groupby("eval_window_id")["bomb_exploded_round"].max().astype(bool).astype(np.int8)
        base = base.merge(exploded.rename("target_BOMB_EXPLODED"), on="eval_window_id", how="left")
    if "has_awp_any_pov" in base.columns:
        base["target_HAS_AWP"] = base["has_awp_any_pov"].fillna(False).astype(bool).astype(np.int8)
    elif "has_awp_any_pov" in windows.columns:
        awp = windows.groupby("eval_window_id")["has_awp_any_pov"].max().astype(bool).astype(np.int8)
        base = base.merge(awp.rename("target_HAS_AWP"), on="eval_window_id", how="left")
    if "round_kills_total" in base.columns:
        base["target_HIGH_KILL_ROUND"] = (base["round_kills_total"].fillna(0).astype(float) >= 5.0).astype(np.int8)
    if "is_pistol_round" in base.columns:
        base["target_PISTOL_ROUND"] = base["is_pistol_round"].fillna(False).astype(bool).astype(np.int8)
    return base


def merge_targets(windows_with_embeddings: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    target_cols = [c for c in targets.columns if c.startswith("target_")]
    keep = ["eval_window_id", *target_cols]
    out = windows_with_embeddings.merge(targets[keep], on="eval_window_id", how="inner")
    if out.empty:
        raise RuntimeError("no embedded windows matched global targets")
    return out


def build_examples(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    target_cols: list[str],
    *,
    split: str,
    k_povs: int,
    subsets_per_window: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    features = []
    targets = []
    meta = []
    split_df = rows[rows["split"].astype(str) == split].copy()
    for eval_window_id, group in split_df.groupby("eval_window_id", sort=False):
        group = group.sort_values("pov_idx").reset_index(drop=True)
        row_ids = group["embedding_row_id"].to_numpy(dtype=np.int64)
        finite = np.isfinite(embeddings[row_ids]).all(axis=1)
        group = group.loc[finite].reset_index(drop=True)
        row_ids = group["embedding_row_id"].to_numpy(dtype=np.int64)
        if len(row_ids) == 0:
            continue
        kk = min(int(k_povs), len(row_ids))
        if kk <= 0:
            continue
        y = group[target_cols].iloc[0].to_numpy(dtype=np.float32)
        for subset_idx in range(int(subsets_per_window)):
            if kk == len(row_ids):
                chosen = np.arange(len(row_ids), dtype=np.int64)
            elif subset_idx == 0:
                chosen = np.arange(kk, dtype=np.int64)
            else:
                chosen = np.sort(rng.choice(np.arange(len(row_ids)), size=kk, replace=False))
            x = embeddings[row_ids[chosen]].astype(np.float32)
            features.append(aggregate_features(x))
            targets.append(y)
            meta.append({
                "eval_window_id": str(eval_window_id),
                "match_id": str(group["match_id"].iloc[0]),
                "round_id": str(group["round_id"].iloc[0]),
                "split": str(split),
                "map_slug": str(group["map_slug"].iloc[0]),
                "phase_bucket": str(group.get("phase_bucket", pd.Series([""])).iloc[0]),
                "k_povs": int(kk),
                "subset_idx": int(subset_idx),
                "selected_povs": ",".join(str(int(v)) for v in group.loc[chosen, "pov_idx"].tolist()),
            })
    if not features:
        return (
            np.empty((0, embeddings.shape[1] * 3), dtype=np.float32),
            np.empty((0, len(target_cols)), dtype=np.float32),
            pd.DataFrame(meta),
        )
    return np.stack(features), np.stack(targets), pd.DataFrame(meta)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 0), dtype=np.float32)
    logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def metrics_from_arrays(y: np.ndarray, p: np.ndarray, target_cols: list[str]) -> dict[str, float]:
    out = {}
    aucs = []
    aps = []
    for idx, target in enumerate(target_cols):
        auc = binary_auc(y[:, idx], p[:, idx])
        ap = binary_average_precision(y[:, idx], p[:, idx])
        prevalence = float(y[:, idx].mean()) if len(y) else float("nan")
        out[f"{target}/auc"] = auc
        out[f"{target}/ap"] = ap
        out[f"{target}/prevalence"] = prevalence
        if np.isfinite(auc):
            aucs.append(float(auc))
        if np.isfinite(ap):
            aps.append(float(ap))
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return out


def metrics_from_predictions(pred_df: pd.DataFrame, target_cols: list[str]) -> dict[str, float]:
    if pred_df.empty:
        return {"macro_auc": float("nan"), "macro_ap": float("nan")}
    y = np.stack([pred_df[f"{target}_target"].to_numpy(dtype=np.float32) for target in target_cols], axis=1)
    p = np.stack([pred_df[f"{target}_prob"].to_numpy(dtype=np.float32) for target in target_cols], axis=1)
    return metrics_from_arrays(y, p, target_cols)


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, [(x - mean) / std if len(x) else x for x in others], mean, std


def train_one_k(
    *,
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    target_cols: list[str],
    k_povs: int,
    args,
    device: torch.device,
    wandb_run=None,
    step_offset: int = 0,
) -> tuple[dict, pd.DataFrame, dict]:
    x_train, y_train, _ = build_examples(
        rows, embeddings, target_cols,
        split="train", k_povs=k_povs,
        subsets_per_window=args.train_subsets_per_window,
        seed=args.seed + k_povs,
    )
    x_val, y_val, _ = build_examples(
        rows, embeddings, target_cols,
        split="val", k_povs=k_povs,
        subsets_per_window=args.eval_subsets_per_window,
        seed=args.seed + 1000 + k_povs,
    )
    x_test, y_test, test_df = build_examples(
        rows, embeddings, target_cols,
        split="test", k_povs=k_povs,
        subsets_per_window=args.eval_subsets_per_window,
        seed=args.seed + 2000 + k_povs,
    )
    if len(x_train) == 0:
        raise RuntimeError(f"no train examples for k={k_povs}")
    x_train, (x_val, x_test), mean, std = _standardize(x_train, x_val, x_test)
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = np.divide(neg, np.maximum(pos, 1.0)).astype(np.float32)
    model = MultiPovProbe(x_train.shape[1], len(target_cols), args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
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
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        val_probs = predict(model, x_val, device)
        val_metrics = metrics_from_arrays(y_val, val_probs, target_cols) if len(x_val) else {}
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
        "train": metrics_from_arrays(y_train, train_probs, target_cols),
        "val": metrics_from_arrays(y_val, val_probs, target_cols) if len(x_val) else {},
        "test": metrics_from_arrays(y_test, test_probs, target_cols) if len(x_test) else {},
        "history": history,
    }
    pred_df = test_df.copy()
    for idx, target in enumerate(target_cols):
        pred_df[f"{target}_target"] = y_test[:, idx] if len(y_test) else []
        pred_df[f"{target}_prob"] = test_probs[:, idx] if len(test_probs) else []
    if len(pred_df) and args.bootstrap_samples > 0:
        metric_keys = ["macro_auc", "macro_ap"]
        for target in target_cols:
            metric_keys.extend([f"{target}/auc", f"{target}/ap"])
        ci = cluster_bootstrap(
            pred_df,
            cluster_col="match_id",
            metric_fn=lambda df: metrics_from_predictions(df, target_cols),
            metrics=metric_keys,
            n_boot=args.bootstrap_samples,
            seed=args.seed + 3000 + k_povs,
        )
        metrics["test"].update(ci)
    state = {
        "model_state": model.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "target_cols": target_cols,
        "k_povs": int(k_povs),
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
    }
    return metrics, pred_df, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 10])
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
    add_wandb_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    windows = read_parquet(args.windows)
    targets = build_global_targets(windows, root=args.root)
    available_targets = [c for c in args.targets if c in targets.columns]
    if not available_targets:
        raise ValueError(f"none of the requested targets are available: {args.targets}")
    index, embeddings = load_embedding_table(args.embeddings)
    rows = merge_targets(attach_embedding_rows(windows, index), targets)

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(args, job_type="train-global-event-probe", config=vars(args))
    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    pred_paths = []
    ckpt_paths = []
    for k_order, k_povs in enumerate(args.k_values):
        metrics, pred_df, state = train_one_k(
            rows=rows,
            embeddings=embeddings,
            target_cols=available_targets,
            k_povs=int(k_povs),
            args=args,
            device=device,
            wandb_run=wandb_run,
            step_offset=k_order * (args.epochs + 10),
        )
        all_metrics.append(metrics)
        pred_path = args.out / f"global_event_probe_k{k_povs}_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        pred_paths.append(pred_path)
        ckpt_path = args.out / f"global_event_probe_k{k_povs}.pt"
        torch.save(state, ckpt_path)
        ckpt_paths.append(ckpt_path)
        log_metrics(wandb_run, metrics, prefix=f"k{k_povs}", summary=True)
        if not pred_df.empty:
            log_dataframe(wandb_run, f"k{k_povs}/test_predictions", pred_df.head(500), max_rows=500)

    metrics = {
        "target_cols": available_targets,
        "target_prevalence": {c: float(targets[c].mean()) for c in available_targets},
        "k_values": [int(k) for k in args.k_values],
        "embedding_rows": int(len(index)),
        "embedding_dim": int(embeddings.shape[1]),
        "windows_sha256": dataframe_sha256(windows),
        "targets_sha256": dataframe_sha256(targets),
        "attached_rows": int(len(rows)),
        "results": all_metrics,
        "git_commit": git_commit(),
    }
    metrics_path = args.out / "metrics_global_event_probe.json"
    write_json(metrics_path, metrics)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-global-event-probe",
            artifact_type="model",
            paths=[metrics_path, args.out / "config.json", *pred_paths, *ckpt_paths],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
