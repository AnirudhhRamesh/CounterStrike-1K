"""Compare single, synchronized-10, and cross-round-10 future-event probes.

Two preregistered feature modes are supported:

``input-grouping``
    All arms use one frozen video encoder.  The shuffled arm keeps the same
    anchor POV but replaces the other nine embeddings with same-split,
    different-match distractors.  This tests whether synchronized observations
    contain predictive information before world-model training.

``checkpoint-representations``
    Each arm reads causal context features from its corresponding frozen MIRA
    checkpoint.  Synchronized and cross-round grouped models both receive the
    identical held-out synchronized ten-POV context; the checkpoints differ
    only in their training grouping.  This is the model-level
    synchronized-training control.  ``shuffled`` remains the internal arm key
    for backward compatibility; it never denotes shuffled actions here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cs2_release.action_probe.train_multipov import MultiPovProbe, aggregate_features
from cs2_release.core.embeddings import load_embedding_table
from cs2_release.core.io import (
    dataframe_sha256,
    git_commit,
    read_parquet,
    sha256_file,
    write_json,
)
from cs2_release.core.metrics import binary_auc, binary_average_precision

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


Arm = Literal["single", "synchronized", "shuffled"]
ARMS: tuple[Arm, ...] = ("single", "synchronized", "shuffled")


def _metrics(y: np.ndarray, p: np.ndarray, targets: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    aucs: list[float] = []
    aps: list[float] = []
    for idx, target in enumerate(targets):
        auc = binary_auc(y[:, idx], p[:, idx])
        ap = binary_average_precision(y[:, idx], p[:, idx])
        out[f"{target}/auc"] = auc
        out[f"{target}/ap"] = ap
        out[f"{target}/prevalence"] = float(y[:, idx].mean()) if len(y) else float("nan")
        if np.isfinite(auc):
            aucs.append(float(auc))
        if np.isfinite(ap):
            aps.append(float(ap))
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return out


def _window_groups(
    embedding_index: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict[str, pd.DataFrame]:
    index = embedding_index.copy()
    index["embedding_row_id"] = np.arange(len(index), dtype=np.int64)
    groups: dict[str, pd.DataFrame] = {}
    for window_id, group in index.groupby("eval_window_id", sort=False):
        group = group.sort_values("pov_idx").reset_index(drop=True)
        if group["pov_idx"].astype(int).tolist() != list(range(10)):
            continue
        row_ids = group["embedding_row_id"].to_numpy(dtype=np.int64)
        if not np.isfinite(embeddings[row_ids]).all():
            continue
        groups[str(window_id)] = group
    return groups


def _candidate_windows(
    labels: pd.DataFrame,
    *,
    target: pd.Series,
) -> list[str]:
    candidates = labels[
        (labels["split"].astype(str) == str(target["split"]))
        & (labels["match_id"].astype(str) != str(target["match_id"]))
    ]
    if "phase_bucket" in labels.columns:
        matched_phase = candidates[
            candidates["phase_bucket"].astype(str) == str(target.get("phase_bucket", ""))
        ]
        if len(matched_phase) >= 9:
            candidates = matched_phase
    return candidates["eval_window_id"].astype(str).tolist()


def build_arm_examples(
    labels: pd.DataFrame,
    embedding_index: pd.DataFrame,
    embeddings: np.ndarray,
    target_cols: list[str],
    *,
    split: str,
    arm: Arm,
    seed: int,
    checkpoint_representations: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build deterministic, one-example-per-target-window probe inputs."""

    if arm not in ARMS:
        raise ValueError(f"unsupported arm {arm!r}")
    groups = _window_groups(embedding_index, embeddings)
    split_labels = (
        labels[labels["split"].astype(str) == split]
        .sort_values(["match_id", "round_id", "eval_window_id"])
        .reset_index(drop=True)
    )
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    meta: list[dict] = []
    for _, target in split_labels.iterrows():
        window_id = str(target["eval_window_id"])
        group = groups.get(window_id)
        if group is None:
            continue
        window_digest = hashlib.sha256(f"{seed}\0{window_id}".encode()).digest()
        anchor_pov = int.from_bytes(window_digest[:8], "little") % 10
        rng = np.random.default_rng(int.from_bytes(window_digest[8:16], "little"))
        anchor_row = group[group["pov_idx"].astype(int) == anchor_pov].iloc[0]
        selected_rows = [int(anchor_row["embedding_row_id"])]
        source_windows = [window_id]

        if arm == "synchronized" or (arm == "shuffled" and checkpoint_representations):
            selected_rows = group["embedding_row_id"].astype(int).tolist()
            source_windows = [window_id] * 10
        elif arm == "shuffled":
            candidates = _candidate_windows(labels, target=target)
            candidates = [candidate for candidate in candidates if candidate in groups]
            permutation = rng.permutation(len(candidates)) if candidates else np.empty(0, dtype=int)
            used_rounds = {str(target["round_id"])}
            for pov_idx in range(10):
                if pov_idx == anchor_pov:
                    continue
                chosen_window = None
                for candidate_idx in permutation:
                    candidate = candidates[int(candidate_idx)]
                    candidate_label = labels[
                        labels["eval_window_id"].astype(str) == candidate
                    ].iloc[0]
                    round_id = str(candidate_label["round_id"])
                    if round_id in used_rounds:
                        continue
                    chosen_window = candidate
                    used_rounds.add(round_id)
                    break
                if chosen_window is None:
                    break
                candidate_group = groups[chosen_window]
                candidate_row = candidate_group[
                    candidate_group["pov_idx"].astype(int) == pov_idx
                ].iloc[0]
                selected_rows.append(int(candidate_row["embedding_row_id"]))
                source_windows.append(chosen_window)
                candidates.remove(chosen_window)
                permutation = rng.permutation(len(candidates))
            if len(selected_rows) != 10:
                continue

        x = embeddings[np.asarray(selected_rows, dtype=np.int64)].astype(np.float32)
        features.append(aggregate_features(x))
        targets.append(target[target_cols].to_numpy(dtype=np.float32))
        meta.append(
            {
                "eval_window_id": window_id,
                "match_id": str(target["match_id"]),
                "round_id": str(target["round_id"]),
                "split": split,
                "arm": arm,
                "anchor_pov": anchor_pov,
                "feature_rows": len(selected_rows),
                "source_windows": ",".join(source_windows),
                "source_rounds": len(
                    {
                        str(
                            labels[
                                labels["eval_window_id"].astype(str) == source_window
                            ]["round_id"].iloc[0]
                        )
                        for source_window in source_windows
                    }
                ),
            }
        )
    if not features:
        return (
            np.empty((0, embeddings.shape[1] * 3), dtype=np.float32),
            np.empty((0, len(target_cols)), dtype=np.float32),
            pd.DataFrame(meta),
        )
    return np.stack(features), np.stack(targets), pd.DataFrame(meta)


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    if not len(x):
        return np.empty((0, 0), dtype=np.float32)
    logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _train_arm(
    *,
    arm: Arm,
    seed: int,
    labels: pd.DataFrame,
    embedding_index: pd.DataFrame,
    embeddings: np.ndarray,
    target_cols: list[str],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_representations: bool,
) -> tuple[dict, pd.DataFrame, dict]:
    built = {}
    for split, offset in (("train", 0), ("val", 10_000), ("test", 20_000)):
        built[split] = build_arm_examples(
            labels,
            embedding_index,
            embeddings,
            target_cols,
            split=split,
            arm=arm,
            seed=seed + offset,
            checkpoint_representations=checkpoint_representations,
        )
    x_train, y_train, _ = built["train"]
    x_val, y_val, _ = built["val"]
    x_test, y_test, test_meta = built["test"]
    if not len(x_train) or not len(x_val) or not len(x_test):
        raise RuntimeError(f"{arm}: an experimental split produced no examples")

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MultiPovProbe(
        input_dim=x_train.shape[1],
        output_dim=len(target_cols),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(
            np.divide(neg, np.maximum(pos, 1.0)).astype(np.float32)
        ).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_state = None
    best_score = -np.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        val_metrics = _metrics(y_val, _predict(model, x_val, device), target_cols)
        score = float(val_metrics["macro_ap"])
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_macro_ap": score,
                "elapsed_seconds": float(time.time() - started),
            }
        )
    if best_state is None:
        raise RuntimeError(f"{arm}: validation produced no finite selection score")
    model.load_state_dict(best_state)
    model.eval()
    probabilities = _predict(model, x_test, device)
    predictions = test_meta.copy()
    for idx, target in enumerate(target_cols):
        predictions[f"{target}_target"] = y_test[:, idx]
        predictions[f"{target}_prob"] = probabilities[:, idx]
    state = {
        "model_state": model.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "target_cols": target_cols,
        "arm": arm,
        "seed": seed,
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
    }
    metrics = {
        "arm": arm,
        "seed": seed,
        "examples": {
            split: len(built[split][0]) for split in ("train", "val", "test")
        },
        "best_val_macro_ap": float(best_score),
        "test": _metrics(y_test, probabilities, target_cols),
        "history": history,
    }
    return metrics, predictions, state


def _prediction_metrics(frame: pd.DataFrame, targets: list[str]) -> dict[str, float]:
    y = np.stack(
        [frame[f"{target}_target"].to_numpy(dtype=np.float32) for target in targets],
        axis=1,
    )
    p = np.stack(
        [frame[f"{target}_prob"].to_numpy(dtype=np.float32) for target in targets],
        axis=1,
    )
    return _metrics(y, p, targets)


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    targets: list[str],
    left_name: str,
    right_name: str,
    samples: int,
    seed: int,
) -> dict:
    """Paired match-cluster bootstrap of left-minus-right event metrics."""

    keys = ["eval_window_id", "match_id", "round_id"]
    columns = keys + [
        column
        for target in targets
        for column in (f"{target}_target", f"{target}_prob")
    ]
    merged = left[columns].merge(
        right[columns],
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("paired arms do not contain identical test windows")

    def arm_frame(source: pd.DataFrame, suffix: str) -> pd.DataFrame:
        out = source[keys].copy()
        for target in targets:
            out[f"{target}_target"] = source[f"{target}_target_{suffix}"]
            out[f"{target}_prob"] = source[f"{target}_prob_{suffix}"]
        return out

    left_metrics = _prediction_metrics(arm_frame(merged, "left"), targets)
    right_metrics = _prediction_metrics(arm_frame(merged, "right"), targets)
    metric_names = ["macro_ap", "macro_auc"]
    observed = {
        metric: float(left_metrics[metric] - right_metrics[metric])
        for metric in metric_names
    }
    clusters = np.asarray(sorted(merged["match_id"].astype(str).unique()))
    grouped = {
        match_id: group
        for match_id, group in merged.groupby("match_id", sort=False)
    }
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metric_names}
    for _ in range(samples):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[str(match_id)] for match_id in chosen], ignore_index=True)
        boot_left = _prediction_metrics(arm_frame(boot, "left"), targets)
        boot_right = _prediction_metrics(arm_frame(boot, "right"), targets)
        for metric in metric_names:
            delta = float(boot_left[metric] - boot_right[metric])
            if np.isfinite(delta):
                draws[metric].append(delta)
    return {
        "definition": f"{left_name}_minus_{right_name}",
        "cluster_unit": "match_id",
        "clusters": len(clusters),
        "bootstrap_samples": int(samples),
        "observed": observed,
        "ci95": {
            metric: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            if values
            else [float("nan"), float("nan")]
            for metric, values in draws.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    feature_source = parser.add_mutually_exclusive_group(required=True)
    feature_source.add_argument(
        "--embeddings",
        type=Path,
        help="One frozen encoder table for the input-grouping information control.",
    )
    feature_source.add_argument(
        "--checkpoint-embeddings",
        type=Path,
        help="Directory containing single/, synchronized/, and shuffled/ MIRA feature tables.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--min-train-positives", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = not args.deterministic
    labels = read_parquet(args.labels)
    checkpoint_representations = args.checkpoint_embeddings is not None
    if checkpoint_representations:
        embedding_paths = {
            arm: args.checkpoint_embeddings / arm
            for arm in ARMS
        }
        embedding_tables = {
            arm: load_embedding_table(path)
            for arm, path in embedding_paths.items()
        }
        reference = embedding_tables["synchronized"][0][
            ["eval_window_id", "sample_key", "pov_idx"]
        ].reset_index(drop=True)
        for arm, (index, embeddings) in embedding_tables.items():
            observed = index[["eval_window_id", "sample_key", "pov_idx"]].reset_index(
                drop=True
            )
            if not observed.equals(reference):
                raise ValueError(
                    f"{arm}: checkpoint representation index differs from synchronized"
                )
            if embeddings.shape[1] != embedding_tables["synchronized"][1].shape[1]:
                raise ValueError(f"{arm}: checkpoint representation dimension drifted")
    else:
        embedding_paths = {arm: args.embeddings for arm in ARMS}
        embedding_index, embeddings = load_embedding_table(args.embeddings)
        embedding_tables = {
            arm: (embedding_index, embeddings)
            for arm in ARMS
        }
    requested = args.targets or [
        column for column in labels.columns if column.startswith("target_")
    ]
    missing = sorted(set(requested) - set(labels.columns))
    if missing:
        raise ValueError(f"requested targets are absent: {missing}")
    train = labels[labels["split"].astype(str) == "train"]
    target_cols = []
    excluded = {}
    for target in requested:
        positives = int(train[target].sum())
        negatives = int(len(train) - positives)
        if positives >= args.min_train_positives and negatives >= args.min_train_positives:
            target_cols.append(target)
        else:
            excluded[target] = {
                "train_positives": positives,
                "train_negatives": negatives,
                "reason": "insufficient training support",
            }
    if not target_cols:
        raise ValueError("no targets passed the training-only support threshold")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    comparisons = []
    for seed in args.seeds:
        predictions: dict[str, pd.DataFrame] = {}
        for arm in ARMS:
            arm_index, arm_embeddings = embedding_tables[arm]
            metrics, pred, state = _train_arm(
                arm=arm,
                seed=int(seed),
                labels=labels,
                embedding_index=arm_index,
                embeddings=arm_embeddings,
                target_cols=target_cols,
                args=args,
                device=device,
                checkpoint_representations=checkpoint_representations,
            )
            arm_dir = args.out / f"seed_{seed}" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(arm_dir / "test_predictions.parquet", index=False)
            torch.save(state, arm_dir / "probe.pt")
            write_json(arm_dir / "metrics.json", metrics)
            predictions[arm] = pred
            results.append(metrics)
        for right in ("shuffled", "single"):
            comparisons.append(
                {
                    "seed": int(seed),
                    "left": "synchronized",
                    "right": right,
                    **paired_bootstrap(
                        predictions["synchronized"],
                        predictions[right],
                        targets=target_cols,
                        left_name="synchronized",
                        right_name=right,
                        samples=args.bootstrap_samples,
                        seed=int(seed) + (30_000 if right == "shuffled" else 40_000),
                    ),
                }
            )

    summary = {
        "schema": "cs1k-causal-future-event-probe-v1",
        "experimental_unit": "one context window from one match-disjoint split",
        "feature_mode": (
            "checkpoint-representations"
            if checkpoint_representations
            else "input-grouping"
        ),
        "arms": (
            {
                "single": "one deterministic anchor from the frozen single-MIRA checkpoint",
                "synchronized": (
                    "all ten held-out synchronized POV features from the frozen "
                    "synchronized-trained MIRA checkpoint"
                ),
                "shuffled": (
                    "the identical ten held-out synchronized POVs represented by the frozen "
                    "matched-information cross-round-grouped MIRA checkpoint; "
                    "'shuffled' is only the backward-compatible internal arm key"
                ),
            }
            if checkpoint_representations
            else {
                "single": "one deterministic anchor POV",
                "synchronized": "all ten POVs from the target round and time",
                "shuffled": (
                    "the same anchor plus nine same-split, different-match distractor POVs; "
                    "ten embeddings and identical probe capacity"
                ),
            }
        ),
        "targets": target_cols,
        "excluded_targets": excluded,
        "seeds": [int(seed) for seed in args.seeds],
        "results": results,
        "paired_comparisons": comparisons,
        "labels_sha256": dataframe_sha256(labels),
        "embedding_sources": {
            arm: {
                "path": str(embedding_paths[arm]),
                "embedding_index_sha256": dataframe_sha256(index),
                "embedding_rows": len(index),
                "embedding_dim": int(values.shape[1]),
                "metadata_sha256": (
                    sha256_file(embedding_paths[arm] / "metadata.json")
                    if (embedding_paths[arm] / "metadata.json").is_file()
                    else None
                ),
                "checkpoint_sha256": (
                    json.loads(
                        (embedding_paths[arm] / "metadata.json").read_text(
                            encoding="utf-8"
                        )
                    ).get("checkpoint_sha256")
                    if (embedding_paths[arm] / "metadata.json").is_file()
                    else None
                ),
            }
            for arm, (index, values) in embedding_tables.items()
        },
        "bootstrap_samples": int(args.bootstrap_samples),
        "deterministic_algorithms": bool(args.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_commit": git_commit(),
    }
    write_json(args.out / "summary.json", summary)
    (args.out / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    print(args.out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
