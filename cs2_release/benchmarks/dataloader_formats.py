"""Run the CounterStrike-1K dataloader format benchmark on a matched subset.

This is the release-facing entry point for the data-format microbenchmark used
in the paper. It assumes three local raw roots can be produced from the same
clips:

* per-clip MP4 with the source/default encode,
* per-clip MP4 re-encoded as GOP=32 with B-frames,
* per-clip MP4 re-encoded as GOP=32 with no B-frames.

The script builds a matched subset, packs WebDataset shards for each codec
profile, then runs the same worker sweep across raw MP4 and WebDataset variants.
It writes JSON results, a summary JSON, an SVG plot, and a LaTeX tabular.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _workers_grid(value: str) -> list[int]:
    if value == "powers":
        return [1, 2, 4, 8, 16, 32, 64]
    if value == "dense":
        return list(range(1, 65))
    return [int(x) for x in value.replace(",", " ").split()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-default-root", type=Path, required=True)
    ap.add_argument("--raw-gop32-bframes-root", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--subset-size", type=int, default=512)
    ap.add_argument("--subset-seed", type=int, default=1729)
    ap.add_argument("--workers-grid", default="powers",
                    help="'powers', 'dense', or an explicit list")
    ap.add_argument("--target-seconds", type=float, default=5.0)
    ap.add_argument("--warmup-seconds", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--encode-workers", type=int, default=32)
    ap.add_argument("--target-shard-bytes", type=int, default=4_194_304)
    args = ap.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.work_root / f"data-loader-format-{run_id}"
    subset_root = run_root / "subset"
    results = run_root / "results"
    plots = run_root / "plots"
    workers = _workers_grid(args.workers_grid)
    for path in (subset_root, results, plots):
        path.mkdir(parents=True, exist_ok=True)

    provenance = {
        "run_id": run_id,
        "subset_size": args.subset_size,
        "subset_seed": args.subset_seed,
        "workers_grid": workers,
        "target_seconds": args.target_seconds,
        "warmup_seconds": args.warmup_seconds,
        "repeats": args.repeats,
        "T": args.T,
        "batch_size": args.batch_size,
        "target_shard_bytes": args.target_shard_bytes,
    }
    (results / "provenance.json").write_text(json.dumps(provenance, indent=2))

    _run([
        sys.executable, "-m", "cs2_benchmark.data_loader.make_subset_roots",
        "--raw-default-root", str(args.raw_default_root),
        "--raw-gop32-bframes-root", str(args.raw_gop32_bframes_root),
        "--out-root", str(subset_root),
        "--n", str(args.subset_size),
        "--seed", str(args.subset_seed),
    ])

    raw_default = subset_root / "raw_default"
    raw_gop32_bframes = subset_root / "raw_gop32_bframes"
    raw_release = subset_root / "raw_release"
    _run([
        sys.executable, "-m", "cs2_benchmark.data_loader.prepare_optimized_clips",
        "--src-root", str(raw_default),
        "--dst-root", str(raw_release),
        "--workers", str(args.encode_workers),
        "--profile", "release",
    ])

    shard_dirs = {
        "default": subset_root / "shards_default",
        "gop32_bframes": subset_root / "shards_gop32_bframes",
        "release": subset_root / "shards_release",
    }
    clip_roots = {
        "default": raw_default / "videos",
        "gop32_bframes": raw_gop32_bframes / "videos",
        "release": raw_release / "videos",
    }
    for name, out_dir in shard_dirs.items():
        _run([
            sys.executable, "-m", "cs2_train.scripts.bench.pack_shards",
            "default",
            "--clips-dir", str(clip_roots[name]),
            "--actions-root", str(raw_default / "actions"),
            "--out-dir", str(out_dir),
            "--target-shard-bytes", str(args.target_shard_bytes),
        ])

    _run([
        sys.executable, "-m", "cs2_benchmark.data_loader.bench_6way",
        "--raw-root-default", str(raw_default),
        "--raw-root-gop32-bframes", str(raw_gop32_bframes),
        "--raw-root-release", str(raw_release),
        "--shard-glob-default", str(shard_dirs["default"] / "*.tar"),
        "--shard-glob-gop32-bframes", str(shard_dirs["gop32_bframes"] / "*.tar"),
        "--shard-glob-release", str(shard_dirs["release"] / "*.tar"),
        "--out", str(results / "throughput_6way.json"),
        "--workers-grid", *[str(x) for x in workers],
        "--target-seconds", str(args.target_seconds),
        "--warmup-seconds", str(args.warmup_seconds),
        "--repeats", str(args.repeats),
        "--T", str(args.T),
        "--batch-size", str(args.batch_size),
    ])

    _run([
        sys.executable, "-m", "cs2_benchmark.data_loader.plot_6way",
        "--in-dir", str(results),
        "--out-dir", str(plots),
    ])
    print(f"results: {results}")
    print(f"plots: {plots}")


if __name__ == "__main__":
    main()
