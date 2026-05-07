"""Train a small release checkpoint for 10-POV corruption detection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cs2_release.core.corruption_packs import attach_pack_embedding_rows, build_pack_arrays
from cs2_release.core.embeddings import load_embedding_table
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import binary_accuracy, binary_auc
from cs2_release.core.tracking import add_wandb_args, finish_wandb, log_artifact, log_metrics


class CorruptionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, arrays, device: torch.device) -> dict[str, float]:
    if len(arrays.labels) == 0:
        return {"auc": float("nan"), "accuracy": float("nan")}
    x = torch.from_numpy(arrays.features).to(device)
    logits = model(x).detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    out = {
        "auc": binary_auc(arrays.labels, probs),
        "accuracy": binary_accuracy(arrays.labels, probs),
    }
    for severity in sorted(set(int(s) for s in arrays.severities.tolist())):
        mask = (arrays.severities == severity) | (arrays.labels == 1)
        if mask.sum() == len(mask) and severity == 0:
            continue
        if mask.any():
            out[f"severity_{severity}_auc"] = binary_auc(arrays.labels[mask], probs[mask])
            out[f"severity_{severity}_accuracy"] = binary_accuracy(arrays.labels[mask], probs[mask])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-packs", type=Path, required=True)
    parser.add_argument("--val-packs", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-log-every", type=int, default=10,
                        help="Training steps between W&B speed/loss logs.")
    add_wandb_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    index, embeddings = load_embedding_table(args.embeddings)
    train_packs_raw = read_parquet(args.train_packs)
    train_packs = attach_pack_embedding_rows(train_packs_raw, index)
    train_arrays = build_pack_arrays(train_packs, embeddings)
    if train_arrays.features.size == 0:
        raise RuntimeError("no train packs could be resolved against embeddings")

    val_arrays = None
    val_packs = None
    if args.val_packs is not None:
        val_packs_raw = read_parquet(args.val_packs)
        val_packs = attach_pack_embedding_rows(val_packs_raw, index)
        val_arrays = build_pack_arrays(val_packs, embeddings)

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(
        args,
        job_type="train-corruption-head",
        config={
            **vars(args),
            "device_resolved": str(device),
            "cuda_device_count": torch.cuda.device_count(),
            "embedding_rows": int(len(index)),
            "embedding_dim": int(embeddings.shape[1]),
            "train_examples": int(len(train_arrays.labels)),
            "val_examples": int(len(val_arrays.labels)) if val_arrays is not None else 0,
        },
    )

    model = CorruptionMLP(
        input_dim=int(train_arrays.features.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(
        torch.from_numpy(train_arrays.features),
        torch.from_numpy(train_arrays.labels),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    history = []
    best_auc = -1.0
    best_state = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        epoch_started = time.time()
        step_started = time.time()
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
            if wandb_run is not None and (
                global_step == 1 or global_step % max(1, args.wandb_log_every) == 0
            ):
                elapsed = max(time.time() - step_started, 1e-9)
                examples = int(x.shape[0])
                payload = {
                    "train/loss_step": float(loss.item()),
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "train/batch_examples_per_sec": examples / elapsed,
                    "train/batch_ms": elapsed * 1000.0,
                    "train/lr": float(opt.param_groups[0]["lr"]),
                }
                if device.type == "cuda":
                    payload["train/cuda_allocated_gib"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    payload["train/cuda_reserved_gib"] = torch.cuda.memory_reserved(device) / (1024 ** 3)
                wandb_run.log(payload, step=global_step)
            step_started = time.time()
        model.eval()
        train_metrics = evaluate(model, train_arrays, device)
        val_metrics = evaluate(model, val_arrays, device) if val_arrays is not None else {}
        epoch_s = max(time.time() - epoch_started, 1e-9)
        monitor_auc = float(val_metrics.get("auc", train_metrics["auc"]))
        if monitor_auc > best_auc:
            best_auc = monitor_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epoch_record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "epoch_s": epoch_s,
            "examples_per_sec": float(len(train_arrays.labels) / epoch_s),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        log_metrics(wandb_run, epoch_record, prefix="epoch", step=global_step)

    if best_state is not None:
        model.load_state_dict(best_state)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "corruption_head.pt"
    payload = {
        "model_state": model.state_dict(),
        "input_dim": int(train_arrays.features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "embedding_dir": str(args.embeddings),
    }
    torch.save(payload, ckpt_path)
    final_train = evaluate(model, train_arrays, device)
    final_val = evaluate(model, val_arrays, device) if val_arrays is not None else {}
    metrics = {
        "checkpoint": str(ckpt_path),
        "train": final_train,
        "val": final_val,
        "history": history,
        "train_packs": int(len(train_packs)),
        "train_examples": int(len(train_arrays.labels)),
        "val_packs": int(len(val_packs)) if val_packs is not None else 0,
        "val_examples": int(len(val_arrays.labels)) if val_arrays is not None else 0,
        "train_packs_sha256": dataframe_sha256(train_packs_raw),
        "git_commit": git_commit(),
    }
    write_json(args.out / "metrics_train_corruption.json", metrics)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    log_metrics(wandb_run, metrics, prefix="corruption_train", summary=True)
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-corruption-head",
            artifact_type="model",
            paths=[ckpt_path, args.out / "metrics_train_corruption.json", args.out / "config.json"],
        )
    finish_wandb(wandb_run)
    print(ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
