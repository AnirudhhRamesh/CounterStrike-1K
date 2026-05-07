"""Evaluate a 10-POV corruption detection checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cs2_release.core.corruption_packs import attach_pack_embedding_rows, build_pack_arrays
from cs2_release.core.embeddings import load_embedding_table
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.metrics import binary_accuracy, binary_auc
from cs2_release.corruption.train import CorruptionMLP
from cs2_release.core.tracking import add_wandb_args, finish_wandb, log_artifact, log_dataframe, log_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    add_wandb_args(parser)
    args = parser.parse_args()

    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(args, job_type="eval-corruption", config=vars(args))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    index, embeddings = load_embedding_table(args.embeddings)
    packs_raw = read_parquet(args.packs)
    packs = attach_pack_embedding_rows(packs_raw, index)
    arrays = build_pack_arrays(packs, embeddings)
    if arrays.features.size == 0:
        raise RuntimeError("no packs could be resolved against embeddings")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CorruptionMLP(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(arrays.features).to(device)).detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    metrics = {
        "auc": binary_auc(arrays.labels, probs),
        "accuracy": binary_accuracy(arrays.labels, probs),
        "examples": int(len(arrays.labels)),
        "packs_resolved": int(len(packs)),
        "packs_input": int(len(packs_raw)),
        "checkpoint": str(args.checkpoint),
        "packs_sha256": dataframe_sha256(packs_raw),
        "git_commit": git_commit(),
    }
    for severity in sorted(set(int(s) for s in arrays.severities.tolist())):
        if severity == 0:
            continue
        mask = (arrays.severities == severity) | (arrays.labels == 1)
        metrics[f"severity_{severity}_auc"] = binary_auc(arrays.labels[mask], probs[mask])
        metrics[f"severity_{severity}_accuracy"] = binary_accuracy(arrays.labels[mask], probs[mask])

    out_dir = args.out if args.out.suffix == "" else args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_corruption.json"
    predictions_path = out_dir / "corruption_predictions.parquet"
    write_json(metrics_path, metrics)
    pred_rows = packs[packs["pack_id"].astype(str).isin(set(arrays.pack_ids))].copy()
    score_map = {pack_id: (float(score), float(logit)) for pack_id, score, logit in zip(arrays.pack_ids, probs, logits, strict=True)}
    pred_rows["score"] = pred_rows["pack_id"].astype(str).map(lambda pack_id: score_map[pack_id][0])
    pred_rows["logit"] = pred_rows["pack_id"].astype(str).map(lambda pack_id: score_map[pack_id][1])
    pred_rows.to_parquet(predictions_path, index=False)
    log_metrics(wandb_run, metrics, prefix="corruption_eval", summary=True)
    log_dataframe(
        wandb_run,
        "corruption/predictions",
        pred_rows.sort_values("score", ascending=False),
        max_rows=500,
    )
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name="cs2-corruption-eval",
            artifact_type="eval-results",
            paths=[metrics_path, predictions_path],
        )
    finish_wandb(wandb_run)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
