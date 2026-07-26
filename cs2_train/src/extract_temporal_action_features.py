"""Extract frozen DINOv2 features for the train/validation ARR probe.

The sampling plan is fixed before video decoding, split-restricted, and
independent of any world-model checkpoint or held-out test data. Arrays are
written atomically and hashed; ``metadata.json`` is the completion marker.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import CSDataset
from .rollout_archive import sha256_file
from .temporal_action_probe import (
    ACTION_LABEL_NAMES,
    MOUSE_DIRECTION_THRESHOLD_DEGREES,
    DinoV2FrameEncoder,
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


def select_feature_indices(
    dataset_size: int,
    *,
    num_windows: int,
    seed: int,
) -> list[int]:
    if not 0 < num_windows <= dataset_size:
        raise ValueError(
            f"num_windows must be in [1,{dataset_size}], got {num_windows}"
        )
    rng = np.random.default_rng(seed)
    return (
        np.sort(rng.choice(dataset_size, size=num_windows, replace=False))
        .astype(np.int64)
        .tolist()
    )


def sample_plan_for_indices(
    dataset: CSDataset,
    indices: list[int],
) -> list[dict]:
    plan = []
    for dataset_index in indices:
        clip, local_start, frame_ids = dataset._resolve_window(dataset_index)
        alive_end_frame = clip.get("alive_end_frame")
        if alive_end_frame is None or (
            isinstance(alive_end_frame, (float, np.floating))
            and np.isnan(alive_end_frame)
        ):
            alive_end_frame = clip["num_frames"]
        plan.append(
            {
                "dataset_index": int(dataset_index),
                "sample_key": str(clip.get("sample_key", "")),
                "match_id": str(clip["match_id"]),
                "round_id": str(clip.get("round_id", "")),
                "pov_idx": int(clip.get("pov_idx", clip.get("player_id", 0))),
                "source_start_frame": int(local_start),
                "source_frame_indices": [int(value) for value in frame_ids],
                "alive_end_frame": int(alive_end_frame),
            }
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-name",
        default="manifest_dust2_confirmatory_spatial_v1.parquet",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-windows", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=20250725)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--decode-height", type=int, default=126)
    parser.add_argument("--decode-width", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid data-loader size")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.decode_height <= 0 or args.decode_width <= 0:
        raise ValueError("decode dimensions must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"feature archive destination is not empty: {args.out_dir}"
        )

    torch.manual_seed(args.sampling_seed)
    np.random.seed(args.sampling_seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    dataset = CSDataset(
        data_path=args.data_dir,
        split=args.split,
        T=9,
        target_fps=8,
        resize=(args.decode_height, args.decode_width),
        manifest_name=args.manifest_name,
        mode="dict",
        window_mode="sliding",
        map_slug="dust2",
        resolution="360p",
    )
    indices = select_feature_indices(
        len(dataset),
        num_windows=args.num_windows,
        seed=args.sampling_seed,
    )
    sample_plan = sample_plan_for_indices(dataset, indices)
    sample_plan_path = args.out_dir / "sample_plan.json"
    tmp_sample_plan = args.out_dir / ".sample_plan.tmp.json"
    tmp_sample_plan.write_text(
        json.dumps(sample_plan, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_sample_plan.replace(sample_plan_path)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=False,
    )
    device = torch.device(args.device)
    encoder = DinoV2FrameEncoder(
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    feature_tmp = args.out_dir / ".frame_features_float16.tmp.npy"
    action_tmp = args.out_dir / ".actions_cs2_float32.tmp.npy"
    features = np.lib.format.open_memmap(
        feature_tmp,
        mode="w+",
        dtype=np.float16,
        shape=(args.num_windows, 9, 768),
    )
    actions = np.lib.format.open_memmap(
        action_tmp,
        mode="w+",
        dtype=np.float32,
        shape=(args.num_windows, 8, 14),
    )
    position = 0
    for batch in loader:
        batch_video = batch["video"]
        batch_size = len(batch_video)
        stop = position + batch_size
        features[position:stop] = (
            encoder.encode_video(batch_video).numpy().astype(np.float16)
        )
        actions[position:stop] = (
            batch["actions"][:, :8].float().numpy().astype(np.float32)
        )
        position = stop
    if position != args.num_windows:
        raise ValueError(
            f"feature extractor wrote {position} rows, expected {args.num_windows}"
        )
    features.flush()
    actions.flush()
    del features
    del actions

    feature_path = args.out_dir / "frame_features_float16.npy"
    action_path = args.out_dir / "actions_cs2_float32.npy"
    feature_tmp.replace(feature_path)
    action_tmp.replace(action_path)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "train_or_validate_temporal_cs2_action_probe",
        "split": args.split,
        "num_windows": args.num_windows,
        "segments_per_window": 2,
        "frames_per_segment": 5,
        "transitions_per_segment": 4,
        "segment_duration_seconds": 0.5,
        "sampling": {
            "population_windows": len(dataset),
            "without_replacement": True,
            "seed": args.sampling_seed,
            "selected_indices_sorted": True,
        },
        "dataset": {
            "data_dir": str(args.data_dir),
            "manifest": str(args.data_dir / args.manifest_name),
            "manifest_sha256": sha256_file(args.data_dir / args.manifest_name),
            "map_slug": "dust2",
            "source_resolution": "360p",
            "source_fps": 32,
            "target_fps": 8,
            "window_mode": "sliding_alive_only",
            "decode_resize": [args.decode_height, args.decode_width],
        },
        "labels": {
            "names": list(ACTION_LABEL_NAMES),
            "buttons": "active in any of four transitions",
            "mouse": "signed sum over four transitions",
            "mouse_threshold_degrees": MOUSE_DIRECTION_THRESHOLD_DEGREES,
            "unmapped_excluded": ["INSPECT", "USE"],
        },
        "backbone": encoder.provenance(),
        "software": {
            "release_commit": git_commit(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "deterministic_algorithms": True,
        },
        "artifacts": {
            "sample_plan": {
                "path": sample_plan_path.name,
                "sha256": sha256_file(sample_plan_path),
            },
            "frame_features": {
                "path": feature_path.name,
                "sha256": sha256_file(feature_path),
                "dtype": "float16",
                "shape": [args.num_windows, 9, 768],
                "axes": "sample,frame,feature",
            },
            "actions_cs2": {
                "path": action_path.name,
                "sha256": sha256_file(action_path),
                "dtype": "float32",
                "shape": [args.num_windows, 8, 14],
                "axes": "sample,transition,canonical_cs2_action",
            },
        },
    }
    metadata_path = args.out_dir / "metadata.json"
    tmp_metadata = args.out_dir / ".metadata.tmp.json"
    tmp_metadata.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_metadata.replace(metadata_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
