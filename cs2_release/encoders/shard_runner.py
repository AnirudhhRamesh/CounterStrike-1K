"""Run frozen embedding extraction across multiple GPU processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from cs2_release.encoders.registry import canonical_encoder_name


def row_ranges(total_rows: int, shard_count: int) -> list[tuple[int, int]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    out = []
    for shard_idx in range(shard_count):
        start = (total_rows * shard_idx) // shard_count
        end = (total_rows * (shard_idx + 1)) // shard_count
        out.append((start, end))
    return out


def terminate(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 20
    for proc in processes:
        remaining = max(deadline - time.time(), 0.1)
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--encoder", default="dinov2_vitb14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", nargs="+", default=["0"],
                        help="CUDA_VISIBLE_DEVICES values, one extraction process per entry.")
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--out", type=Path, required=True,
                        help="Final embedding parent directory or exact encoder directory.")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Temporary parent for per-shard embedding directories.")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    total_rows = len(pd.read_parquet(args.windows))
    encoder_name = canonical_encoder_name(args.encoder)
    final_out = args.out
    if final_out.name != encoder_name:
        final_out = final_out / encoder_name
    work_dir = args.work_dir or final_out.parent.parent / "embedding_shards"
    log_dir = args.log_dir or work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[subprocess.Popen] = []
    log_handles = []
    shard_dirs: list[Path] = []
    shard_root = args.shard_root or args.root
    for shard_idx, (start, end) in enumerate(row_ranges(total_rows, len(args.devices))):
        shard_parent = work_dir / f"shard{shard_idx:02d}"
        shard_dir = shard_parent / encoder_name
        shard_dirs.append(shard_dir)
        cmd = [
            args.python, "-m", "cs2_release.encoders.extract_video",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(args.windows),
            "--encoder", args.encoder,
            "--device", args.device,
            "--frames-per-window", str(args.frames_per_window),
            "--row-start", str(start),
            "--row-end", str(end),
            "--out", str(shard_parent),
            "--wandb-mode", "disabled",
            "--wandb-preview-windows", "0",
        ]
        if args.verify_sha256:
            cmd.append("--verify-sha256")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.devices[shard_idx])
        log_path = log_dir / f"shard{shard_idx:02d}.log"
        handle = log_path.open("w", encoding="utf-8")
        log_handles.append(handle)
        print(f"+ CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)} > {log_path}", flush=True)
        processes.append(subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env))

    try:
        while True:
            statuses = [proc.poll() for proc in processes]
            if all(status is not None for status in statuses):
                break
            failed = [status for status in statuses if status not in {None, 0}]
            if failed:
                terminate(processes)
                raise subprocess.CalledProcessError(failed[0], "embedding shard extraction")
            time.sleep(15)
    finally:
        for handle in log_handles:
            handle.close()

    for proc in processes:
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, "embedding shard extraction")

    merge_cmd = [
        args.python, "-m", "cs2_release.encoders.merge",
        "--shards", *[str(path) for path in shard_dirs],
        "--out", str(final_out),
    ]
    print("+ " + " ".join(merge_cmd), flush=True)
    subprocess.run(merge_cmd, check=True)
    print(final_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
