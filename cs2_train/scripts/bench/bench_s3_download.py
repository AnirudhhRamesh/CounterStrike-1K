"""Benchmark S3 download throughput: per-clip mp4 vs WebDataset shards.

Cold-cache, repeated runs, MB/s + wall-clock + per-object timings.

Each test downloads a known set of objects to /opt/dlami/nvme/bench/dl_test/
and removes them after measurement. Page cache is dropped between tests via
`echo 3 > /proc/sys/vm/drop_caches` (requires sudo or fallback flush).

Usage:
    bench_s3_download.py \
        --clips-prefix s3://cs2-counterstrike-1k-dataset-clips-s3/videos/counterstrike-1k-dataset/demo_10f2f0115a508559/player_<...>/ \
        --shards-prefix s3://cs2-counterstrike-1k-dataset-shards-s3/default/ \
        --shards-opt-prefix s3://cs2-counterstrike-1k-dataset-shards-s3/optimized/ \
        --target-bytes 8000000000 \
        --concurrency-grid 1 4 8 16 32 \
        --out /opt/dlami/nvme/bench/results/s3_download.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def list_objects(prefix: str, max_bytes: int) -> tuple[list[tuple[str, int]], int]:
    """List objects under prefix until accumulated bytes ≥ max_bytes.

    Returns (object_uris_with_sizes, total_bytes).
    """
    bucket, key_prefix = parse_s3_uri(prefix)
    cmd = ["aws", "s3api", "list-objects-v2",
           "--bucket", bucket,
           "--prefix", key_prefix,
           "--query", "Contents[].[Key,Size]",
           "--output", "text"]
    out = subprocess.check_output(cmd, text=True)
    rows = []
    total = 0
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        k, s = parts[0], int(parts[1])
        if s == 0:
            continue
        rows.append((f"s3://{bucket}/{k}", s))
        total += s
        if total >= max_bytes:
            break
    return rows, total


def drop_caches() -> bool:
    """Drop linux page cache. Returns True if successful."""
    try:
        subprocess.check_call(["sudo", "-n", "sh", "-c",
                               "sync && echo 3 > /proc/sys/vm/drop_caches"],
                              stderr=subprocess.DEVNULL)
        return True
    except Exception:  # noqa: BLE001
        return False


def time_sync_concurrent(uris: list[str], dest_dir: Path, concurrency: int) -> dict:
    """Spawn `concurrency` parallel `aws s3 cp` processes pulling chunks of uris.

    Returns dict with wall_s, n_objects, bytes, mb_per_s.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Write the URI list to a tmp file per worker
    chunks = [[] for _ in range(concurrency)]
    for i, u in enumerate(uris):
        chunks[i % concurrency].append(u)

    procs = []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        list_path = dest_dir / f"_list_{i}.txt"
        list_path.write_text("\n".join(chunk))
        # We use a small bash loop that calls aws s3 cp per line — keeps the
        # 'connection reuse' that boto3 would give us in a single client.
        cmd = ["bash", "-c",
               f"while read u; do "
               f"  fname=$(basename \"$u\"); "
               f"  aws s3 cp --quiet \"$u\" {dest_dir}/$$_${{fname}}; "
               f"done < {list_path}"]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    wall = time.time() - t0
    bytes_pulled = sum(p.stat().st_size for p in dest_dir.glob("*") if p.is_file())
    return {
        "wall_s":    round(wall, 3),
        "n_objects": len(uris),
        "bytes":     bytes_pulled,
        "mb_per_s":  round(bytes_pulled / 1e6 / max(wall, 1e-6), 2),
    }


def time_sync_native(prefix: str, dest_dir: Path, concurrency: int = 10) -> dict:
    """Use `aws s3 sync` with default concurrency (its own thread pool)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AWS_S3_MAX_CONCURRENT_REQUESTS"] = str(concurrency)
    t0 = time.time()
    subprocess.check_call(
        ["aws", "s3", "sync", "--quiet", prefix, str(dest_dir)],
        env=env,
    )
    wall = time.time() - t0
    bytes_pulled = 0
    n_objects = 0
    for f in dest_dir.rglob("*"):
        if f.is_file():
            bytes_pulled += f.stat().st_size
            n_objects += 1
    return {
        "wall_s":    round(wall, 3),
        "n_objects": n_objects,
        "bytes":     bytes_pulled,
        "mb_per_s":  round(bytes_pulled / 1e6 / max(wall, 1e-6), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-prefix", required=True,
                    help="S3 URI prefix for individual mp4 clips")
    ap.add_argument("--shards-prefix", required=True,
                    help="S3 URI prefix for default WebDataset shards")
    ap.add_argument("--shards-opt-prefix", default=None,
                    help="(optional) S3 URI prefix for optimized shards")
    ap.add_argument("--target-bytes", type=int, default=8_000_000_000,
                    help="How much data to pull per test (default 8 GB)")
    ap.add_argument("--concurrency-grid", nargs="+", type=int,
                    default=[1, 4, 8, 16, 32])
    ap.add_argument("--scratch", default="/opt/dlami/nvme/bench/dl_test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scratch = Path(args.scratch)
    out: dict = {"target_bytes": args.target_bytes, "tests": {}}

    test_specs = [
        ("clips_per_object", args.clips_prefix),
        ("shards_default",   args.shards_prefix),
    ]
    if args.shards_opt_prefix:
        test_specs.append(("shards_optimized", args.shards_opt_prefix))

    for name, prefix in test_specs:
        print(f"\n=== {name} === ({prefix})")
        objs, total = list_objects(prefix, args.target_bytes)
        print(f"Listed {len(objs)} objects, total {total/1e9:.2f} GB")
        out["tests"][name] = {
            "prefix":     prefix,
            "n_objects":  len(objs),
            "list_bytes": total,
            "results":    {},
        }
        uris = [u for u, _ in objs]

        for c in args.concurrency_grid:
            shutil.rmtree(scratch, ignore_errors=True)
            cache_dropped = drop_caches()
            print(f"  concurrency={c}, cache_dropped={cache_dropped} ...")
            r = time_sync_concurrent(uris, scratch, c)
            r["cache_dropped"] = cache_dropped
            out["tests"][name]["results"][str(c)] = r
            print(f"    wall={r['wall_s']}s, mb_per_s={r['mb_per_s']}, "
                  f"objects={r['n_objects']}")

        # Native sync as a sanity baseline (uses aws cli's default thread pool)
        shutil.rmtree(scratch, ignore_errors=True)
        cache_dropped = drop_caches()
        r = time_sync_native(prefix, scratch, concurrency=10)
        r["cache_dropped"] = cache_dropped
        out["tests"][name]["results"]["aws_sync_default"] = r
        print(f"    aws_sync_default: wall={r['wall_s']}s, mb_per_s={r['mb_per_s']}")

    shutil.rmtree(scratch, ignore_errors=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
