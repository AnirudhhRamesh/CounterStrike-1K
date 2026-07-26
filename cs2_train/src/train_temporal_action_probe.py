"""Train the MIRA-style temporal CS2 action-recoverability probe.

Only frozen DINOv2 feature archives from the train and validation splits are
accepted. Model selection uses validation macro average precision; no test
split or world-model rollout is read.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cs2_release.core.metrics import binary_auc

from .rollout_archive import sha256_file
from .temporal_action_probe import (
    ACTION_LABEL_NAMES,
    MOUSE_DIRECTION_THRESHOLD_DEGREES,
    TemporalActionProbe,
    actions_to_labels,
    binary_average_precision_tie_aware,
)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_feature_archive(path: Path, *, expected_split: str) -> dict:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("status") != "complete"
        or metadata.get("split") != expected_split
    ):
        raise ValueError(f"{metadata_path}: incompatible feature archive")
    for name, artifact in metadata["artifacts"].items():
        artifact_path = path / artifact["path"]
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"{name}: feature-archive SHA-256 mismatch")
        if artifact_path.suffix == ".npy":
            array = np.load(artifact_path, mmap_mode="r")
            if (
                list(array.shape) != artifact["shape"]
                or str(array.dtype) != artifact["dtype"]
            ):
                raise ValueError(f"{name}: feature-archive array contract mismatch")
    return metadata


class ProbeSegmentDataset(Dataset):
    def __init__(self, archive_dir: Path, metadata: dict) -> None:
        artifacts = metadata["artifacts"]
        self.features = np.load(
            archive_dir / artifacts["frame_features"]["path"],
            mmap_mode="r",
        )
        self.actions = np.load(
            archive_dir / artifacts["actions_cs2"]["path"],
            mmap_mode="r",
        )
        if len(self.features) != len(self.actions):
            raise ValueError("feature/action row counts differ")

    def __len__(self) -> int:
        return 2 * len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = index // 2
        start = 4 * (index % 2)
        features = torch.from_numpy(
            np.array(
                self.features[sample, start : start + 5],
                dtype=np.float32,
                copy=True,
            )
        )
        actions = torch.from_numpy(
            np.array(
                self.actions[sample, start : start + 4],
                dtype=np.float32,
                copy=True,
            )
        )
        return features, actions_to_labels(actions)


def probe_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    per_class = {}
    aps = []
    aucs = []
    for index, name in enumerate(ACTION_LABEL_NAMES):
        ap = binary_average_precision_tie_aware(
            labels[:, index],
            probabilities[:, index],
        )
        auc = binary_auc(labels[:, index], probabilities[:, index])
        prevalence = float(labels[:, index].mean())
        per_class[name] = {
            "average_precision": ap,
            "auroc": auc,
            "prevalence": prevalence,
            "positives": int(labels[:, index].sum()),
            "examples": len(labels),
        }
        if np.isfinite(ap):
            aps.append(ap)
        if np.isfinite(auc):
            aucs.append(auc)
    return {
        "macro_average_precision": (float(np.mean(aps)) if aps else float("nan")),
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "per_class": per_class,
    }


@torch.inference_mode()
def evaluate_probe(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    for features, target in loader:
        logits = model(features.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.numpy())
    y = np.concatenate(labels).astype(np.float32)
    p = np.concatenate(probabilities).astype(np.float32)
    return probe_metrics(y, p), y, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20250725)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch size and epochs must be positive")
    if args.lr <= 0 or args.max_pos_weight <= 0:
        raise ValueError("learning rate and max positive weight must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError(f"probe output directory is not empty: {args.out_dir}")

    train_metadata = verify_feature_archive(
        args.train_features,
        expected_split="train",
    )
    val_metadata = verify_feature_archive(
        args.val_features,
        expected_split="val",
    )
    compatibility_fields = (
        ("dataset", "manifest_sha256"),
        ("dataset", "map_slug"),
        ("dataset", "source_resolution"),
        ("dataset", "source_fps"),
        ("dataset", "target_fps"),
        ("dataset", "decode_resize"),
        ("labels", "names"),
        ("labels", "mouse_threshold_degrees"),
        ("backbone", "repository"),
        ("backbone", "model"),
        ("backbone", "weights_sha256"),
        ("backbone", "preprocess"),
    )
    for section, key in compatibility_fields:
        if train_metadata[section][key] != val_metadata[section][key]:
            raise ValueError(f"train/validation feature drift at {section}.{key}")
    train_plan = json.loads(
        (
            args.train_features / train_metadata["artifacts"]["sample_plan"]["path"]
        ).read_text(encoding="utf-8")
    )
    val_plan = json.loads(
        (
            args.val_features / val_metadata["artifacts"]["sample_plan"]["path"]
        ).read_text(encoding="utf-8")
    )
    train_matches = {str(row["match_id"]) for row in train_plan}
    val_matches = {str(row["match_id"]) for row in val_plan}
    overlap = train_matches & val_matches
    if overlap:
        raise ValueError(
            "train/validation action-probe feature archives overlap by match: "
            f"{sorted(overlap)[:3]}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    train_dataset = ProbeSegmentDataset(args.train_features, train_metadata)
    val_dataset = ProbeSegmentDataset(args.val_features, val_metadata)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_actions = torch.from_numpy(
        np.array(train_dataset.actions, dtype=np.float32, copy=True)
    )
    all_train_labels = (
        torch.stack(
            (
                actions_to_labels(train_actions[:, 0:4]),
                actions_to_labels(train_actions[:, 4:8]),
            ),
            dim=1,
        )
        .reshape(-1, len(ACTION_LABEL_NAMES))
        .numpy()
    )
    positive = all_train_labels.sum(axis=0)
    negative = len(all_train_labels) - positive
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError(
            "every action-probe class needs positive and negative examples"
        )
    pos_weight = np.minimum(
        negative / positive,
        args.max_pos_weight,
    ).astype(np.float32)

    model = TemporalActionProbe(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics, _, _ = evaluate_probe(model, val_loader, device=device)
        score = float(val_metrics["macro_average_precision"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val": val_metrics,
            }
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("validation macro AP was never finite")
    model.load_state_dict(best_state)
    train_metrics, _, _ = evaluate_probe(
        model,
        train_eval_loader,
        device=device,
    )
    val_metrics, val_labels, val_probabilities = evaluate_probe(
        model,
        val_loader,
        device=device,
    )

    checkpoint = {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "label_names": list(ACTION_LABEL_NAMES),
        "label_contract": train_metadata["labels"],
        "backbone": train_metadata["backbone"],
        "best_epoch": best_epoch,
        "selection_metric": "validation_macro_average_precision",
        "selection_value": best_score,
        "seed": args.seed,
        "train_feature_metadata_sha256": sha256_file(
            args.train_features / "metadata.json"
        ),
        "val_feature_metadata_sha256": sha256_file(args.val_features / "metadata.json"),
        "release_commit": git_commit(),
    }
    checkpoint_path = args.out_dir / "temporal_action_probe.pt"
    checkpoint_tmp = args.out_dir / ".temporal_action_probe.tmp.pt"
    torch.save(checkpoint, checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)
    prediction_path = args.out_dir / "validation_predictions.npz"
    prediction_tmp = args.out_dir / ".validation_predictions.tmp.npz"
    with prediction_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=val_labels,
            probabilities=val_probabilities,
        )
    prediction_tmp.replace(prediction_path)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "mira_style_temporal_cs2_action_recoverability_probe",
        "backbone_deviation_from_mira": (
            "MIRA uses gated DINOv3-B; this reproducible public evaluator uses "
            "commit-pinned DINOv2-B/14."
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_predictions": str(prediction_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "model_config": model.config(),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "selection_metric": "validation_macro_average_precision",
        "selection_value": best_score,
        "positive_weights": {
            name: float(pos_weight[index])
            for index, name in enumerate(ACTION_LABEL_NAMES)
        },
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "train": train_metrics,
        "val": val_metrics,
        "history": history,
        "train_feature_metadata": str(args.train_features / "metadata.json"),
        "train_feature_metadata_sha256": sha256_file(
            args.train_features / "metadata.json"
        ),
        "val_feature_metadata": str(args.val_features / "metadata.json"),
        "val_feature_metadata_sha256": sha256_file(args.val_features / "metadata.json"),
        "label_names": list(ACTION_LABEL_NAMES),
        "mouse_threshold_degrees": MOUSE_DIRECTION_THRESHOLD_DEGREES,
        "software": {
            "release_commit": git_commit(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "deterministic_algorithms": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_tmp = args.out_dir / ".summary.tmp.json"
    summary_tmp.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
