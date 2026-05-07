"""Extract per-window log-mel audio features for CounterStrike-1K evaluations.

Mirrors :mod:`cs2_release.encoders.extract_video`. Produces a directory layout
identical to the video embedding output so downstream probes can swap encoders
without changes:

    <out>/audio_logmel/
        embedding_index.parquet  # one row per eval window, with
                                 #   eval_window_id, sample_key, pov_idx, ...
        audio_features.npz       # arrays:
                                 #   features      (N, n_mels, n_time) float32
                                 #   eval_window_id (N,) str
                                 #   sample_key    (N,) str
                                 #   pov_idx       (N,) int16
        metadata.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cs2_release.core.audio import decode_audio_window, waveform_to_log_mel
from cs2_release.core.io import (
    DatasetRoots,
    dataframe_sha256,
    git_commit,
    read_parquet,
    read_video_bytes,
    write_json,
)


def _sample_index_for(roots: DatasetRoots) -> pd.DataFrame | None:
    for candidate in (
        roots.root / f"sample_index_{roots.resolution}.parquet",
        roots.root / "sample_index.parquet",
    ):
        if candidate.exists():
            return read_parquet(candidate)
    return None


def extract_audio_features(
    *,
    windows_path: Path,
    roots: DatasetRoots,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmin: float,
    fmax: float | None,
    max_windows: int | None,
    row_start: int | None,
    row_end: int | None,
    verify_sha256: bool,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    windows = read_parquet(windows_path).reset_index(drop=True).copy()
    windows["embedding_source_row_id"] = np.arange(len(windows), dtype=np.int64)
    if row_start is not None or row_end is not None:
        start = 0 if row_start is None else int(row_start)
        end = len(windows) if row_end is None else int(row_end)
        windows = windows.iloc[start:end].reset_index(drop=True)
    if max_windows is not None:
        windows = windows.head(int(max_windows)).copy()

    sample_index = _sample_index_for(roots)
    if "fps" not in windows.columns:
        raise ValueError("eval_windows.parquet missing fps column")
    if "frame_tick_stride" not in windows.columns:
        raise ValueError("eval_windows.parquet missing frame_tick_stride column")

    features: list[np.ndarray] = []
    failed: list[dict] = []
    feature_shape: tuple[int, int] | None = None
    started = time.time()

    for _, row in tqdm(windows.iterrows(), total=len(windows), desc="audio windows"):
        sample_key = str(row["sample_key"])
        try:
            fps = float(row["fps"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            start_seconds = start_frame / fps
            duration_seconds = max(end_frame - start_frame, 1) / fps
            video_bytes = read_video_bytes(
                sample_key,
                roots=roots,
                sample_index=sample_index,
                verify_sha256=verify_sha256,
            )
            waveform = decode_audio_window(
                video_bytes,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                sample_rate=int(sample_rate),
            )
            mel = waveform_to_log_mel(
                waveform,
                sample_rate=int(sample_rate),
                n_mels=int(n_mels),
                n_fft=int(n_fft),
                hop_length=int(hop_length),
                fmin=float(fmin),
                fmax=fmax,
            )
            if feature_shape is None:
                feature_shape = mel.shape
            elif mel.shape != feature_shape:
                # Pad/trim to the canonical time length so the npz is rectangular.
                target_t = feature_shape[1]
                if mel.shape[1] >= target_t:
                    mel = mel[:, :target_t]
                else:
                    pad = np.full((mel.shape[0], target_t - mel.shape[1]),
                                  mel.min() if mel.size else -6.0,
                                  dtype=np.float32)
                    mel = np.concatenate([mel, pad], axis=1)
            features.append(mel.astype(np.float32, copy=False))
        except Exception as exc:  # noqa: BLE001 - record and keep the run reproducible.
            failed.append({
                "sample_key": sample_key,
                "eval_window_id": str(row["eval_window_id"]),
                "error": repr(exc),
            })
            features.append(None)  # type: ignore[arg-type]

    if feature_shape is None:
        # No windows decoded successfully; choose canonical shape from params.
        canonical_t = 1 + int(round((sample_rate / float(windows["fps"].iloc[0]) * 1.0) / hop_length)) if len(windows) else 1
        feature_shape = (int(n_mels), max(1, canonical_t))

    # Failed windows get NaN so downstream `_split_arrays` filters them via the
    # `np.isfinite(...)` check rather than silently training on fill values.
    nan_mat = np.full(feature_shape, np.nan, dtype=np.float32)
    arrays = [
        nan_mat if mel is None else mel
        for mel in features
    ]
    feats = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    info = {
        "feature_shape": list(feature_shape),
        "failed": failed,
        "elapsed_s": round(time.time() - started, 3),
    }
    return windows, feats, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--row-start", type=int, default=None)
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--encoder-name", default="audio_logmel",
                        help="Subdirectory under --out used to namespace this feature set.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roots = DatasetRoots.from_args(root=args.root, shard_root=args.shard_root, resolution=args.resolution)
    windows, features, info = extract_audio_features(
        windows_path=args.windows,
        roots=roots,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        max_windows=args.max_windows,
        row_start=args.row_start,
        row_end=args.row_end,
        verify_sha256=args.verify_sha256,
    )

    out_dir = args.out / args.encoder_name if args.out.suffix == "" else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "embedding_index.parquet"
    npz_path = out_dir / "audio_features.npz"
    windows.to_parquet(index_path, index=False)
    np.savez_compressed(
        npz_path,
        features=features,
        sample_key=windows["sample_key"].astype(str).to_numpy(),
        eval_window_id=windows["eval_window_id"].astype(str).to_numpy(),
        pov_idx=windows["pov_idx"].to_numpy(dtype=np.int16),
        embedding_source_row_id=windows["embedding_source_row_id"].to_numpy(dtype=np.int64),
    )
    metadata = {
        "encoder": args.encoder_name,
        "feature_shape": info["feature_shape"],
        "rows": int(len(windows)),
        "failed": info["failed"],
        "failed_count": int(len(info["failed"])),
        "sample_rate": int(args.sample_rate),
        "n_mels": int(args.n_mels),
        "n_fft": int(args.n_fft),
        "hop_length": int(args.hop_length),
        "fmin": float(args.fmin),
        "fmax": None if args.fmax is None else float(args.fmax),
        "row_start": args.row_start,
        "row_end": args.row_end,
        "resolution": args.resolution,
        "windows_path": str(args.windows),
        "windows_sha256": dataframe_sha256(windows),
        "elapsed_s": info["elapsed_s"],
        "git_commit": git_commit(),
    }
    write_json(out_dir / "metadata.json", metadata)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
