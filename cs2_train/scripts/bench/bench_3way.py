"""Three-way dataloader benchmark: per-clip mp4 vs default shards vs optimized shards.

Builds the same DataLoader sweep as run_bench but across all three formats,
sharing the warmup / measurement window, and emits one combined JSON.

Usage:
    bench_3way.py \
        --raw-root          /opt/dlami/nvme/bench/clips_raw \
        --shard-glob-default   '/opt/dlami/nvme/bench/shards_default/*.tar' \
        --shard-glob-optimized '/opt/dlami/nvme/bench/shards_optimized/*.tar' \
        --out /opt/dlami/nvme/bench/results/throughput_3way.json \
        --workers-grid 0 4 8 16 32 \
        --target-seconds 22

Note: clips_raw must contain a manifest.json compatible with CSDataset
(produce one with prepare_manifest.py if pulling raw clips fresh from S3).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_train.scripts.bench.run_bench import (   # noqa: E402
    CSDataset, WdsClipDataset, _collate, _measure_loader,
)


def _make_raw_loader(raw_root: Path, T: int, batch_size: int, num_workers: int):
    ds = CSDataset(data_path=raw_root, split="train", T=T, stride=1, mode="dict")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_collate,
    )


def _make_wds_loader(shard_glob: str, T: int, batch_size: int, num_workers: int):
    shards = sorted(glob.glob(shard_glob))
    ds = WdsClipDataset(shards, T=T, stride=1, shuffle_buffer=200)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_collate,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--shard-glob-default",   required=True)
    ap.add_argument("--shard-glob-optimized", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers-grid", nargs="+", type=int,
                    default=[0, 4, 8, 16, 32])
    ap.add_argument("--target-seconds", type=float, default=22.0)
    args = ap.parse_args()

    torch.multiprocessing.set_start_method("forkserver", force=True)
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    out: dict = {
        "T": args.T,
        "batch_size": args.batch_size,
        "target_seconds": args.target_seconds,
        "shard_count_default":   len(sorted(glob.glob(args.shard_glob_default))),
        "shard_count_optimized": len(sorted(glob.glob(args.shard_glob_optimized))),
        "results": {
            "raw_mp4":  {},
            "wds_default":   {},
            "wds_optimized": {},
        },
    }

    fmts = [
        ("raw_mp4",       lambda nw: _make_raw_loader(args.raw_root, args.T, args.batch_size, nw)),
        ("wds_default",   lambda nw: _make_wds_loader(args.shard_glob_default,   args.T, args.batch_size, nw)),
        ("wds_optimized", lambda nw: _make_wds_loader(args.shard_glob_optimized, args.T, args.batch_size, nw)),
    ]

    for nw in args.workers_grid:
        for fmt_name, mk in fmts:
            print(f"[3way] fmt={fmt_name} workers={nw}")
            try:
                loader = mk(nw)
                # warm-up
                _measure_loader(loader, target_seconds=3.0, max_batches=4)
                # measure
                r = _measure_loader(loader, target_seconds=args.target_seconds)
                out["results"][fmt_name][str(nw)] = r
                print(f"  {fmt_name:>14}/w={nw}: {r['samples_per_s']:.2f} sps, "
                      f"{r['ms_per_batch']:.1f} ms/batch")
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR {fmt_name} w={nw}: {e!r}")
                out["results"][fmt_name][str(nw)] = {"error": repr(e)}
            finally:
                del loader

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
