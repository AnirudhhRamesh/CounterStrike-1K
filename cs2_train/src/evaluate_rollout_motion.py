"""Score archived world-model rollouts by matched optical-flow dynamics.

This is a model-agnostic complement to pixel MSE.  A fixed torchvision
RAFT-Small estimator extracts motion from the generated and ground-truth
trajectories.  We compare the resulting flow fields only over the central
world-view crop, excluding most of the static HUD and first-person weapon.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .rollout_archive import sha256_file

PRIMARY_METRICS = (
    "flow_epe_normalized_mean",
    "global_flow_epe_normalized_mean",
)


def verify_archive(archive_dir: Path) -> dict:
    metadata_path = archive_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("status") != "complete":
        raise ValueError(f"{metadata_path}: unsupported or incomplete archive")
    for name, artifact in metadata["artifacts"].items():
        path = archive_dir / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"{name}: rollout archive SHA-256 mismatch")
        array = np.load(path, mmap_mode="r")
        if list(array.shape) != artifact["shape"] or str(array.dtype) != artifact["dtype"]:
            raise ValueError(f"{name}: rollout archive array contract mismatch")
    return metadata


def central_world_crop(
    height: int,
    width: int,
    *,
    top_fraction: float,
    bottom_fraction: float,
    side_fraction: float,
) -> tuple[slice, slice]:
    if not (
        0 <= top_fraction < bottom_fraction <= 1
        and 0 <= side_fraction < 0.5
    ):
        raise ValueError("invalid normalized world-view crop")
    top = round(height * top_fraction)
    bottom = round(height * bottom_fraction)
    side = round(width * side_fraction)
    if bottom <= top or width - side <= side:
        raise ValueError("world-view crop is empty")
    return slice(top, bottom), slice(side, width - side)


def flow_agreement_per_step(
    predicted_flow: torch.Tensor,
    ground_truth_flow: torch.Tensor,
    *,
    crop: tuple[slice, slice],
) -> dict[str, torch.Tensor]:
    """Return full-field and robust-global flow errors for ``[B,T,2,H,W]``."""

    if predicted_flow.shape != ground_truth_flow.shape:
        raise ValueError("predicted and ground-truth flow shapes differ")
    if predicted_flow.ndim != 5 or predicted_flow.shape[2] != 2:
        raise ValueError(f"expected [B,T,2,H,W] flow, got {predicted_flow.shape}")
    y, x = crop
    predicted = predicted_flow[..., y, x]
    target = ground_truth_flow[..., y, x]
    diagonal = math.hypot(predicted_flow.shape[-2], predicted_flow.shape[-1])
    epe = torch.linalg.vector_norm(predicted - target, dim=2).mean(dim=(-2, -1))
    predicted_global = predicted.flatten(-2).median(dim=-1).values
    target_global = target.flatten(-2).median(dim=-1).values
    global_epe = torch.linalg.vector_norm(predicted_global - target_global, dim=2)
    magnitude_error = (
        torch.linalg.vector_norm(predicted_global, dim=2)
        - torch.linalg.vector_norm(target_global, dim=2)
    ).abs()
    return {
        "flow_epe_normalized": epe / diagonal,
        "global_flow_epe_normalized": global_epe / diagonal,
        "global_flow_magnitude_error_normalized": magnitude_error / diagonal,
    }


def masked_sample_means(
    values: torch.Tensor,
    valid_steps: np.ndarray,
) -> tuple[list[list[float | None]], list[float]]:
    valid = torch.from_numpy(np.array(valid_steps, dtype=np.bool_, copy=True)).to(
        values.device
    )
    if values.shape != valid.shape:
        raise ValueError(f"metric/mask shapes differ: {values.shape} vs {valid.shape}")
    if not valid.any(dim=1).all():
        raise ValueError("every rollout must have at least one valid dynamics step")
    per_step = [
        [
            float(value) if bool(is_valid) else None
            for value, is_valid in zip(row.tolist(), mask.tolist(), strict=True)
        ]
        for row, mask in zip(values, valid, strict=True)
    ]
    means = [
        float(row[mask].mean().item())
        for row, mask in zip(values, valid, strict=True)
    ]
    return per_step, means


def paired_round_bootstrap(
    rows: list[dict],
    *,
    metric: str,
    reference: str = "true",
    bootstrap_seed: int,
    n_bootstrap: int,
) -> dict:
    by_key: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    round_for_key: dict[tuple[int, int], str] = {}
    for row in rows:
        key = (int(row["eval_seed"]), int(row["sample_position"]))
        by_key[key][str(row["action_mode"])] = float(row[metric])
        round_for_key[key] = str(row["round_id"])
    output = {}
    modes = sorted({str(row["action_mode"]) for row in rows} - {reference})
    for mode_index, mode in enumerate(modes):
        deltas_by_round: dict[str, list[float]] = defaultdict(list)
        for key, values in by_key.items():
            if {reference, mode}.issubset(values):
                deltas_by_round[round_for_key[key]].append(
                    values[mode] - values[reference]
                )
        round_means = np.asarray(
            [np.mean(values) for values in deltas_by_round.values()],
            dtype=np.float64,
        )
        if round_means.size == 0:
            raise ValueError(f"no paired rows for {metric}/{mode}")
        rng = np.random.default_rng(bootstrap_seed + mode_index)
        draws = rng.integers(
            0,
            round_means.size,
            size=(n_bootstrap, round_means.size),
        )
        bootstrap = round_means[draws].mean(axis=1)
        output[mode] = {
            "delta_definition": f"{mode}_minus_{reference}",
            "interpretation": "positive means true actions match ground-truth motion better",
            "mean_delta": float(round_means.mean()),
            "cluster_unit": "round_id",
            "num_round_clusters": int(round_means.size),
            "bootstrap_seed": bootstrap_seed + mode_index,
            "bootstrap_replicates": n_bootstrap,
            "ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "fraction_rounds_positive": float((round_means > 0).mean()),
        }
    return output


class RaftSmallFlow:
    """Pinned torchvision RAFT-Small C_T_V2 inference wrapper."""

    def __init__(
        self,
        *,
        device: torch.device,
        height: int,
        width: int,
        pair_batch_size: int,
    ) -> None:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        if height % 8 or width % 8 or min(height, width) < 128:
            raise ValueError("RAFT resize must be divisible by 8 and at least 128 px")
        self.weights = Raft_Small_Weights.C_T_V2
        self.transforms = self.weights.transforms()
        self.model = raft_small(weights=self.weights, progress=False).to(device).eval()
        self.device = device
        self.height = int(height)
        self.width = int(width)
        self.pair_batch_size = int(pair_batch_size)

    @torch.inference_mode()
    def sequence_flow(self, video_uint8: np.ndarray) -> torch.Tensor:
        """Estimate adjacent flow for ``[B,T+1,H,W,3]`` uint8 video."""

        video = torch.from_numpy(np.asarray(video_uint8)).permute(0, 1, 4, 2, 3)
        first = video[:, :-1].reshape(-1, 3, video.shape[-2], video.shape[-1])
        second = video[:, 1:].reshape(-1, 3, video.shape[-2], video.shape[-1])
        outputs = []
        for start in range(0, len(first), self.pair_batch_size):
            first_batch = F.interpolate(
                first[start : start + self.pair_batch_size].float().div(255),
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            second_batch = F.interpolate(
                second[start : start + self.pair_batch_size].float().div(255),
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            first_batch, second_batch = self.transforms(first_batch, second_batch)
            outputs.append(
                self.model(
                    first_batch.to(self.device),
                    second_batch.to(self.device),
                )[-1].cpu()
            )
        flow = torch.cat(outputs)
        return flow.reshape(
            video.shape[0],
            video.shape[1] - 1,
            2,
            self.height,
            self.width,
        )

    def provenance(self) -> dict:
        from urllib.parse import urlparse

        import torchvision

        cache_path = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / Path(urlparse(self.weights.url).path).name
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"cannot hash the loaded RAFT checkpoint at {cache_path}"
            )
        return {
            "implementation": "torchvision.models.optical_flow.raft_small",
            "weights": "Raft_Small_Weights.C_T_V2",
            "weights_url": self.weights.url,
            "weights_sha256": sha256_file(cache_path),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "resize": [self.height, self.width],
            "resize_mode": "bilinear_align_corners_false",
            "resize_antialias": False,
            "input_range_before_weight_transforms": "[0,1]",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--flow-pair-batch-size", type=int, default=64)
    parser.add_argument("--flow-height", type=int, default=128)
    parser.add_argument("--flow-width", type=int, default=224)
    parser.add_argument("--crop-top", type=float, default=0.10)
    parser.add_argument("--crop-bottom", type=float, default=0.68)
    parser.add_argument("--crop-side", type=float, default=0.04)
    parser.add_argument("--bootstrap-seed", type=int, default=20250727)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.sample_batch_size <= 0 or args.flow_pair_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    metadata = verify_archive(args.archive_dir)
    artifacts = metadata["artifacts"]
    predictions = np.load(
        args.archive_dir / artifacts["predictions_uint8"]["path"],
        mmap_mode="r",
    )
    ground_truth = np.load(
        args.archive_dir / artifacts["ground_truth_uint8"]["path"],
        mmap_mode="r",
    )
    context_last = np.load(
        args.archive_dir / artifacts["context_last_uint8"]["path"],
        mmap_mode="r",
    )
    valid_steps = np.load(
        args.archive_dir / artifacts["valid_steps_bool"]["path"],
        mmap_mode="r",
    )
    sample_plan_path = args.archive_dir.parent / "sample_plan.json"
    if sha256_file(sample_plan_path) != metadata["contract"]["sample_plan_sha256"]:
        raise ValueError("rollout archive sample-plan hash mismatch")
    sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))

    num_samples = len(sample_plan)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    if predictions.shape[2] < num_samples or ground_truth.shape[0] < num_samples:
        raise ValueError("rollout arrays contain fewer rows than the sample plan")
    device = torch.device(args.device)
    estimator = RaftSmallFlow(
        device=device,
        height=args.flow_height,
        width=args.flow_width,
        pair_batch_size=args.flow_pair_batch_size,
    )
    crop = central_world_crop(
        args.flow_height,
        args.flow_width,
        top_fraction=args.crop_top,
        bottom_fraction=args.crop_bottom,
        side_fraction=args.crop_side,
    )
    eval_seeds = metadata["contract"]["eval_seeds"]
    action_modes = metadata["contract"]["action_modes"]
    rows: list[dict] = []
    for start in range(0, num_samples, args.sample_batch_size):
        stop = min(start + args.sample_batch_size, num_samples)
        positions = list(range(start, stop))
        target_video = np.concatenate(
            [context_last[start:stop, None], ground_truth[start:stop]],
            axis=1,
        )
        target_flow = estimator.sequence_flow(target_video)
        for seed_index, eval_seed in enumerate(eval_seeds):
            for mode_index, action_mode in enumerate(action_modes):
                generated_video = np.concatenate(
                    [
                        context_last[start:stop, None],
                        predictions[seed_index, mode_index, start:stop],
                    ],
                    axis=1,
                )
                generated_flow = estimator.sequence_flow(generated_video)
                metrics = flow_agreement_per_step(
                    generated_flow,
                    target_flow,
                    crop=crop,
                )
                summarized = {
                    name: masked_sample_means(value, valid_steps[start:stop])
                    for name, value in metrics.items()
                }
                for local_index, position in enumerate(positions):
                    info = sample_plan[position]
                    row = {
                        "sample_position": position,
                        "dataset_index": int(info["dataset_index"]),
                        "sample_key": info["sample_key"],
                        "match_id": info["match_id"],
                        "round_id": info["round_id"],
                        "pov_idx": int(info["pov_idx"]),
                        "eval_seed": int(eval_seed),
                        "action_mode": action_mode,
                        "valid_steps": valid_steps[position].tolist(),
                    }
                    for name, (per_step, means) in summarized.items():
                        row[f"{name}_per_step"] = per_step[local_index]
                        row[f"{name}_mean"] = means[local_index]
                    rows.append(row)

    out_dir = args.out_dir or args.archive_dir.parent / "motion_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "per_sample_motion_metrics.jsonl"
    tmp_rows = out_dir / ".per_sample_motion_metrics.tmp.jsonl"
    summary_path = out_dir / "summary.json"
    tmp_summary = out_dir / ".summary.tmp.json"
    occupied = [
        path
        for path in (rows_path, tmp_rows, summary_path, tmp_summary)
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            f"motion-metric destination already contains {occupied[0]}"
        )
    with tmp_rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_rows.replace(rows_path)

    metric_names = tuple(f"{name}_mean" for name in metrics)
    means = {
        mode: {
            metric: float(
                np.mean(
                    [
                        row[metric]
                        for row in rows
                        if row["action_mode"] == mode
                    ]
                )
            )
            for metric in metric_names
        }
        for mode in action_modes
    }
    summary = {
        "schema_version": 1,
        "status": "complete" if args.max_samples is None else "debug_subset",
        "purpose": "paired_ground_truth_optical_flow_dynamics",
        "archive_metadata": str(args.archive_dir / "metadata.json"),
        "archive_metadata_sha256": sha256_file(args.archive_dir / "metadata.json"),
        "sample_plan_sha256": sha256_file(sample_plan_path),
        "num_samples": num_samples,
        "num_rounds": len(
            {sample_plan[index]["round_id"] for index in range(num_samples)}
        ),
        "eval_seeds": eval_seeds,
        "action_modes": action_modes,
        "flow_estimator": estimator.provenance(),
        "world_view_crop": {
            "top_fraction": args.crop_top,
            "bottom_fraction": args.crop_bottom,
            "side_fraction": args.crop_side,
            "pixel_slices": {
                "y": [crop[0].start, crop[0].stop],
                "x": [crop[1].start, crop[1].stop],
            },
        },
        "deterministic_algorithms": True,
        "means": means,
        "paired_deltas": {
            metric: paired_round_bootstrap(
                rows,
                metric=metric,
                bootstrap_seed=args.bootstrap_seed + metric_index * 100,
                n_bootstrap=args.bootstrap_replicates,
            )
            for metric_index, metric in enumerate(PRIMARY_METRICS)
        },
        "per_sample_metrics": str(rows_path),
        "per_sample_metrics_sha256": sha256_file(rows_path),
    }
    tmp_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp_summary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
