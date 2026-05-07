"""Pack rendered mp4 clips into WebDataset tar shards for benchmarking.

Two modes:

  default   — pack source mp4s as-is (whatever GOP/bf the renderer chose).
  optimized — re-encode each clip with the CounterStrike-1K release profile
              (-c:v libx264 -crf 20 -g 32 -bf 0 -fps_mode cfr -movflags +faststart)
              or a named benchmark ablation profile before packing.

Each WebDataset sample is a basename triple:
    {key}.mp4         — the (possibly re-encoded) video
    {key}.actions.parquet — synthetic per-frame action stream (zero-filled
                            unless --actions-from is given)
    {key}.json        — metadata record

Shards target ~1 GB each. We pack samples in their iteration order, switching to
a new shard once the current shard exceeds the target size.

Usage:
    pack_shards.py default \
        --clips-dir /opt/dlami/nvme/bench/clips_raw \
        --out-dir   /opt/dlami/nvme/bench/shards_default \
        --target-shard-bytes 1073741824

    pack_shards.py optimized \
        --clips-dir /opt/dlami/nvme/bench/clips_raw \
        --out-dir   /opt/dlami/nvme/bench/shards_optimized \
        --tmp-dir   /opt/dlami/nvme/bench/clips_opt \
        --workers   16
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ACTION_COLS = [
    "FORWARD", "BACK", "LEFT", "RIGHT", "JUMP", "DUCK", "WALK",
    "FIRE", "RIGHTCLICK", "RELOAD", "INSPECT", "USE",
    "delta_pitch", "delta_yaw",
]

ENCODE_PROFILES: dict[str, list[str]] = {
    "release": [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-g", "32", "-keyint_min", "32", "-bf", "0", "-refs", "1",
        "-fps_mode", "cfr", "-movflags", "+faststart",
        "-x264-params", "scenecut=0:open_gop=0",
        "-pix_fmt", "yuv420p",
    ],
    "gop32_bframes": [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-g", "32", "-keyint_min", "32", "-bf", "2", "-refs", "1",
        "-fps_mode", "cfr", "-movflags", "+faststart",
        "-x264-params", "scenecut=0:open_gop=0",
        "-pix_fmt", "yuv420p",
    ],
}


def discover_clips(clips_dir: Path) -> list[dict]:
    """Walk clips_dir and return one record per mp4 found.

    Produces records with keys: demo_id, player_id, clip_id, mp4_path.
    """
    out = []
    for demo_dir in sorted(clips_dir.glob("demo_*")):
        demo_id = demo_dir.name.replace("demo_", "")
        for player_dir in sorted(demo_dir.glob("player_*")):
            player_id = player_dir.name.replace("player_", "")
            for mp4 in sorted(player_dir.glob("clip_*.mp4")):
                clip_id = int(mp4.stem.replace("clip_", ""))
                out.append({
                    "demo_id":    demo_id,
                    "player_id":  player_id,
                    "clip_id":    clip_id,
                    "mp4_path":   str(mp4),
                })
    return out


def reencode_one(args: tuple) -> tuple[str, bool, str]:
    """Re-encode one clip with a benchmark codec profile.

    Accepts (src, dst) for the legacy 2-tuple form (audio kept), (src, dst,
    audio_mode), or (src, dst, audio_mode, profile). ``profile`` is one of
    ``release`` or ``gop32_bframes``.
    Returns (src, ok, err).
    """
    if len(args) == 4:
        src, dst, audio_mode, profile = args
    elif len(args) == 3:
        src, dst, audio_mode = args
        profile = "release"
    else:
        src, dst = args
        audio_mode = "keep"
        profile = "release"
    if profile not in ENCODE_PROFILES:
        return (src, False, f"unknown encode profile: {profile}")
    audio_args = ["-an"] if audio_mode == "strip" else ["-c:a", "copy"]
    dst_path = Path(dst)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return (src, True, "skip")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst_path.with_name(
        f".{dst_path.stem}.tmp.{os.getpid()}{dst_path.suffix}"
    )
    if tmp_dst.exists():
        tmp_dst.unlink()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-threads", "1",
        "-i", src,
        *ENCODE_PROFILES[profile],
        *audio_args,
        "-threads", "1",
        str(tmp_dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            if tmp_dst.exists():
                tmp_dst.unlink()
            return (src, False, proc.stderr[-500:])
        if not tmp_dst.exists() or tmp_dst.stat().st_size == 0:
            if tmp_dst.exists():
                tmp_dst.unlink()
            return (src, False, "ffmpeg produced empty output")
        os.replace(tmp_dst, dst_path)
        return (src, True, "ok")
    except subprocess.TimeoutExpired:
        if tmp_dst.exists():
            tmp_dst.unlink()
        return (src, False, "timeout")


def fake_actions_parquet(n_frames: int) -> bytes:
    """Build a per-frame actions parquet matching the locked CS2-WM schema.

    Columns: frame_idx + 12 binary buttons + 2 mouse deltas.
    """
    rng = np.random.default_rng(seed=0)
    n_buttons = len(ACTION_COLS) - 2  # last two are mouse
    buttons = rng.integers(0, 2, size=(n_frames, n_buttons), dtype=np.int8).astype(bool)
    mouse = rng.normal(0, 1.5, size=(n_frames, 2)).astype(np.float32)

    df_data = {"frame_idx": np.arange(n_frames, dtype=np.int64)}
    for i, col in enumerate(ACTION_COLS[:n_buttons]):
        df_data[col] = buttons[:, i]
    df_data["delta_pitch"] = mouse[:, 0]
    df_data["delta_yaw"]   = mouse[:, 1]

    table = pa.Table.from_pandas(pd.DataFrame(df_data))
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def actions_parquet_for_clip(args, clip: dict, n_frames: int) -> bytes:
    """Return action sidecar bytes for a packed clip.

    When ``--actions-root`` is set, reuse the already-materialized per-clip
    Parquet so raw MP4 and WDS benchmarks pay for identical sidecar payloads.
    Otherwise fall back to deterministic synthetic actions.
    """
    actions_root = getattr(args, "actions_root", None)
    if actions_root is not None:
        rel = Path(clip["mp4_path"]).relative_to(args.clips_dir)
        actions_path = Path(actions_root) / rel.with_suffix(".parquet")
        if actions_path.is_file():
            return actions_path.read_bytes()
    return fake_actions_parquet(n_frames)


def probe_frames(mp4_path: str) -> int:
    """Return frame count via ffprobe metadata (fast — no decode).

    Reads `nb_frames` from the stream header. May be off by one for streams
    with PTS-rounding artefacts; downstream code handles that with a safety
    margin. ~5 ms per call vs ~1.5 s for -count_frames.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        mp4_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=15).strip()
        n = int(out) if out and out != "N/A" else 0
        if n > 0:
            return n
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        pass
    # Fallback: derive from duration × fps if nb_frames is missing
    cmd2 = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        mp4_path,
    ]
    try:
        out2 = subprocess.check_output(cmd2, text=True, timeout=15).strip().splitlines()
        if len(out2) == 2:
            num, den = out2[0].split("/")
            fps = float(num) / float(den)
            duration = float(out2[1])
            return int(round(duration * fps))
    except Exception:  # noqa: BLE001
        pass
    return 0


def pack(args, clips: list[dict], out_dir: Path, mp4_lookup: dict[str, str]):
    """Pack `clips` into 1-GB tar shards under out_dir.

    mp4_lookup maps src_mp4_path → actual_mp4_path_to_pack (re-encode-aware).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = args.target_shard_bytes

    shard_idx = 0
    sample_idx = 0
    cur_path = out_dir / f"shard-{shard_idx:05d}.tar"
    cur_tar = tarfile.open(cur_path, "w")
    cur_bytes = 0
    n_in_shard = 0
    shard_summaries: list[dict] = []
    t0 = time.time()

    for clip in clips:
        sample_idx += 1
        key = f"{clip['demo_id']}__p{clip['player_id']}__c{clip['clip_id']:03d}"
        src_mp4 = mp4_lookup.get(clip["mp4_path"], clip["mp4_path"])
        if not Path(src_mp4).exists():
            continue
        mp4_size = Path(src_mp4).stat().st_size
        if mp4_size == 0:
            continue

        n_frames = probe_frames(src_mp4)
        if n_frames < 2:
            continue

        # 1. mp4
        info = tarfile.TarInfo(name=f"{key}.mp4")
        info.size = mp4_size
        info.mtime = int(time.time())
        with open(src_mp4, "rb") as f:
            cur_tar.addfile(info, f)

        # 2. actions parquet
        act_bytes = actions_parquet_for_clip(args, clip, n_frames)
        info_a = tarfile.TarInfo(name=f"{key}.actions.parquet")
        info_a.size = len(act_bytes)
        info_a.mtime = int(time.time())
        cur_tar.addfile(info_a, io.BytesIO(act_bytes))

        # 3. json metadata
        meta = {
            "key":        key,
            "demo_id":    clip["demo_id"],
            "player_id":  clip["player_id"],
            "clip_id":    clip["clip_id"],
            "frames":     n_frames,
            "fps":        30.0,
            "duration_s": n_frames / 30.0,
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        info_j = tarfile.TarInfo(name=f"{key}.json")
        info_j.size = len(meta_bytes)
        info_j.mtime = int(time.time())
        cur_tar.addfile(info_j, io.BytesIO(meta_bytes))

        cur_bytes += mp4_size + len(act_bytes) + len(meta_bytes)
        n_in_shard += 1

        # Roll over
        if cur_bytes >= target:
            cur_tar.close()
            shard_summaries.append({
                "shard": cur_path.name,
                "n_samples": n_in_shard,
                "bytes": cur_path.stat().st_size,
            })
            print(f"  wrote {cur_path.name}: {n_in_shard} samples, "
                  f"{cur_path.stat().st_size / 1e9:.2f} GB")
            shard_idx += 1
            cur_path = out_dir / f"shard-{shard_idx:05d}.tar"
            cur_tar = tarfile.open(cur_path, "w")
            cur_bytes = 0
            n_in_shard = 0

    cur_tar.close()
    if cur_path.stat().st_size > 0:
        shard_summaries.append({
            "shard": cur_path.name,
            "n_samples": n_in_shard,
            "bytes": cur_path.stat().st_size,
        })
        print(f"  wrote {cur_path.name}: {n_in_shard} samples, "
              f"{cur_path.stat().st_size / 1e9:.2f} GB")

    summary = {
        "n_shards": len(shard_summaries),
        "total_bytes": sum(s["bytes"] for s in shard_summaries),
        "total_samples": sum(s["n_samples"] for s in shard_summaries),
        "elapsed_s": round(time.time() - t0, 1),
        "shards": shard_summaries,
    }
    (out_dir / "shards.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPacked {summary['n_shards']} shards, "
          f"{summary['total_bytes']/1e9:.2f} GB, "
          f"{summary['total_samples']} samples, "
          f"{summary['elapsed_s']}s.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--clips-dir", required=True, type=Path)
    common_args.add_argument("--out-dir", required=True, type=Path)
    common_args.add_argument("--target-shard-bytes", type=int, default=1_073_741_824)
    common_args.add_argument("--limit", type=int, default=None,
                             help="Cap number of clips packed (debug)")
    common_args.add_argument("--actions-root", type=Path, default=None,
                             help="Optional actions tree mirroring clips-dir")

    p_default = sub.add_parser("default", parents=[common_args],
                               help="Pack source mp4s as-is")
    p_opt = sub.add_parser("optimized", parents=[common_args],
                           help="Re-encode then pack")
    p_opt.add_argument("--tmp-dir", required=True, type=Path,
                       help="Where to stage re-encoded mp4s")
    p_opt.add_argument("--workers", type=int, default=16,
                       help="Parallel ffmpeg workers")
    p_opt.add_argument("--profile", choices=sorted(ENCODE_PROFILES),
                       default="release",
                       help="Benchmark encode profile for optimized mode")

    args = ap.parse_args()

    clips = discover_clips(args.clips_dir)
    if args.limit is not None:
        clips = clips[: args.limit]
    print(f"Discovered {len(clips)} clips under {args.clips_dir}")

    if args.mode == "default":
        mp4_lookup = {c["mp4_path"]: c["mp4_path"] for c in clips}
    else:
        # Re-encode in parallel.
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        mp4_lookup = {}
        for c in clips:
            src = c["mp4_path"]
            rel = Path(src).relative_to(args.clips_dir)
            dst = args.tmp_dir / rel
            jobs.append((src, str(dst)))
            mp4_lookup[src] = str(dst)
        print(f"Re-encoding {len(jobs)} clips with {args.workers} workers "
              f"(profile={args.profile})...")
        n_ok = n_skip = n_err = 0
        t0 = time.time()
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            jobs_with_profile = [(src, dst, "keep", args.profile) for src, dst in jobs]
            for src, ok, msg in ex.map(reencode_one, jobs_with_profile, chunksize=1):
                if ok and msg == "skip":
                    n_skip += 1
                elif ok:
                    n_ok += 1
                else:
                    n_err += 1
                    if n_err <= 5:
                        print(f"  ERR {src}: {msg[:200]}")
                done = n_ok + n_skip + n_err
                if done % 50 == 0:
                    print(f"  re-encoded {done}/{len(jobs)} "
                          f"in {time.time() - t0:.0f}s")
        print(f"Re-encode: {n_ok} ok, {n_skip} skipped, {n_err} errors, "
              f"{time.time() - t0:.0f}s total.")

    print(f"\nPacking shards into {args.out_dir} ...")
    pack(args, clips, args.out_dir, mp4_lookup)


if __name__ == "__main__":
    main()
