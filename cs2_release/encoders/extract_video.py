"""Extract frozen video-window embeddings for CounterStrike-1K evaluations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from cs2_release.core.io import (
    DatasetRoots,
    dataframe_sha256,
    git_commit,
    read_parquet,
    read_video_bytes,
    write_json,
)
from cs2_release.core.tracking import (
    add_wandb_args,
    finish_wandb,
    log_artifact,
    log_images,
    log_metrics,
)
from cs2_release.core.video import (
    decode_sampled_frames,
    make_frame_grid,
    sample_frame_indices,
)
from cs2_release.encoders.registry import build_encoder


def extract_embeddings(
    *,
    windows_path: Path,
    roots: DatasetRoots,
    encoder_name: str,
    device: str,
    frames_per_window: int,
    encoder_batch_size: int,
    max_windows: int | None,
    row_start: int | None,
    row_end: int | None,
    verify_sha256: bool,
    wandb_run=None,
    wandb_preview_windows: int = 0,
) -> tuple[object, np.ndarray, object]:
    windows = read_parquet(windows_path)
    windows = windows.reset_index(drop=True).copy()
    windows["embedding_source_row_id"] = np.arange(len(windows), dtype=np.int64)
    if row_start is not None or row_end is not None:
        start = 0 if row_start is None else int(row_start)
        end = len(windows) if row_end is None else int(row_end)
        windows = windows.iloc[start:end].reset_index(drop=True)
    if max_windows is not None:
        windows = windows.head(max_windows).copy()
    preview_ids = set(
        windows["eval_window_id"].astype(str).drop_duplicates().head(wandb_preview_windows).tolist()
    )
    preview_frames: dict[str, dict[int, np.ndarray]] = {window_id: {} for window_id in preview_ids}
    sample_index = None
    for candidate in (
        roots.root / f"sample_index_{roots.resolution}.parquet",
        roots.root / "sample_index.parquet",
    ):
        if candidate.exists():
            sample_index = read_parquet(candidate)
            break

    encoder = build_encoder(encoder_name, device=device)
    embeddings = np.full((len(windows), encoder.spec.dim), np.nan, dtype=np.float32)
    failed: list[dict] = []
    started = time.time()
    processed_rows = 0
    sample_groups = list(windows.groupby("sample_key", sort=False))
    for sample_key_value, group in tqdm(sample_groups, total=len(sample_groups), desc="embed clips"):
        sample_key = str(sample_key_value)
        try:
            requested = [
                sample_frame_indices(
                    int(row["start_frame"]),
                    int(row["end_frame"]),
                    frames_per_window,
                )
                for _, row in group.iterrows()
            ]
            union_indices = sorted({frame for indices in requested for frame in indices})
            video_bytes = read_video_bytes(
                sample_key,
                roots=roots,
                sample_index=sample_index,
                verify_sha256=verify_sha256,
            )
            frames = decode_sampled_frames(
                video_bytes,
                union_indices,
                resize=encoder.spec.resize,
            )
            frame_position = {frame_idx: position for position, frame_idx in enumerate(union_indices)}
            frame_groups = [
                [frame_position[frame_idx] for frame_idx in indices]
                for indices in requested
            ]
            group_embeddings = encoder.encode_many(
                frames,
                frame_groups,
                batch_size=encoder_batch_size,
            )
            embeddings[group.index.to_numpy(dtype=np.int64)] = group_embeddings
            for (_, row), indices in zip(group.iterrows(), requested, strict=True):
                window_id = str(row["eval_window_id"])
                if window_id in preview_frames:
                    middle_frame = indices[len(indices) // 2]
                    preview_frames[window_id][int(row["pov_idx"])] = frames[
                        frame_position[middle_frame]
                    ]
        except Exception as exc:  # noqa: BLE001 - record and keep batch reproducible.
            for _, row in group.iterrows():
                failed.append({
                    "sample_key": sample_key,
                    "eval_window_id": str(row["eval_window_id"]),
                    "error": repr(exc),
                })
        processed_rows += len(group)
        if wandb_run is not None and (
            processed_rows == len(windows) or processed_rows // 50 != (processed_rows - len(group)) // 50
        ):
            elapsed = max(time.time() - started, 1e-9)
            wandb_run.log({
                "extract/processed_rows": processed_rows,
                "extract/failed_rows": len(failed),
                "extract/rows_per_sec": processed_rows / elapsed,
            })
    previews = []
    for window_id, by_pov in preview_frames.items():
        if not by_pov:
            continue
        povs = sorted(by_pov)
        grid = make_frame_grid(
            [by_pov[pov] for pov in povs],
            labels=[f"pov {pov}" for pov in povs],
            columns=5,
        )
        previews.append((grid, f"{window_id} ({len(povs)} POVs)"))
    return windows, embeddings, {"encoder": encoder, "failed": failed, "previews": previews}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--encoder", default="rgb_hist",
                        help="rgb_hist, torchvision_resnet18, torchvision_resnet50, dinov2_vits14, ...")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=128,
        help="Maximum decoded frames per neural encoder forward pass.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--row-start", type=int, default=None)
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wandb-preview-windows", type=int, default=2,
                        help="Number of eval windows to log as 10-POV frame grids.")
    add_wandb_args(parser)
    args = parser.parse_args()

    roots = DatasetRoots.from_args(root=args.root, shard_root=args.shard_root, resolution=args.resolution)
    from cs2_release.core.tracking import init_wandb

    wandb_run = init_wandb(args, job_type="extract-video-embeddings", config=vars(args))
    started = time.time()
    windows, embeddings, extra = extract_embeddings(
        windows_path=args.windows,
        roots=roots,
        encoder_name=args.encoder,
        device=args.device,
        frames_per_window=args.frames_per_window,
        encoder_batch_size=args.encoder_batch_size,
        max_windows=args.max_windows,
        row_start=args.row_start,
        row_end=args.row_end,
        verify_sha256=args.verify_sha256,
        wandb_run=wandb_run,
        wandb_preview_windows=args.wandb_preview_windows,
    )
    encoder = extra["encoder"]
    failed = extra["failed"]

    out_dir = args.out / encoder.spec.name if args.out.suffix == "" else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "embedding_index.parquet"
    npz_path = out_dir / "embeddings.npz"
    windows.to_parquet(index_path, index=False)
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        sample_key=windows["sample_key"].astype(str).to_numpy(),
        eval_window_id=windows["eval_window_id"].astype(str).to_numpy(),
        pov_idx=windows["pov_idx"].to_numpy(dtype=np.int16),
        embedding_source_row_id=windows["embedding_source_row_id"].to_numpy(dtype=np.int64),
    )
    metadata = {
        "encoder": encoder.spec.name,
        "encoder_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "rows": int(len(windows)),
        "failed": failed,
        "failed_count": int(len(failed)),
        "frames_per_window": int(args.frames_per_window),
        "encoder_batch_size": int(args.encoder_batch_size),
        "row_start": args.row_start,
        "row_end": args.row_end,
        "resolution": args.resolution,
        "windows_path": str(args.windows),
        "windows_sha256": dataframe_sha256(windows),
        "elapsed_s": round(time.time() - started, 3),
        "git_commit": git_commit(),
    }
    write_json(out_dir / "metadata.json", metadata)
    log_metrics(wandb_run, metadata, prefix="extract", summary=True)
    log_images(wandb_run, "preview/10pov_frame_grids", extra["previews"])
    if args.wandb_log_artifacts:
        log_artifact(
            wandb_run,
            name=f"cs2-eval-embeddings-{encoder.spec.name}",
            artifact_type="embeddings",
            paths=[index_path, npz_path, out_dir / "metadata.json"],
        )
    finish_wandb(wandb_run)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
