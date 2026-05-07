"""For a local dir of clips downloaded from S3, synthesize actions parquets
and a CSDataset-compatible manifest.

Layout produced (compatible with cs2_train.src.dataset.CSDataset):
    <root>/
      manifest.json
      videos/demo_X/player_Y/clip_NNN.mp4    (already there from S3 sync)
      actions/demo_X/player_Y/clip_NNN.parquet (synthesized)

Usage:
    build_local_manifest.py \
        --clips-root /opt/dlami/nvme/bench/clips_raw \
        --out-root   /opt/dlami/nvme/bench/raw_mp4 \
        --workers    16
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
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


def probe_frames(mp4: str) -> int:
    """Fast metadata-only frame count. ~5 ms vs ~1.5 s for -count_frames."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        mp4,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10).strip()
        n = int(out) if out and out != "N/A" else 0
        if n > 0:
            return n
    except Exception:  # noqa: BLE001
        pass
    # Fallback: derive from duration × fps
    cmd2 = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        mp4,
    ]
    try:
        out2 = subprocess.check_output(cmd2, text=True, timeout=10).strip().splitlines()
        if len(out2) == 2:
            num, den = out2[0].split("/")
            fps = float(num) / float(den)
            return int(round(float(out2[1]) * fps))
    except Exception:  # noqa: BLE001
        pass
    return 0


def synth_one(args: tuple) -> tuple[str, int, str]:
    """Synthesize a parquet for one clip. Returns (mp4, n_frames, status)."""
    src_mp4, dst_mp4, dst_parquet = args
    if not Path(dst_mp4).exists():
        Path(dst_mp4).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src_mp4, dst_mp4)
    if Path(dst_parquet).exists() and Path(dst_parquet).stat().st_size > 0:
        n = probe_frames(dst_mp4)
        return (src_mp4, n, "skip")

    n = probe_frames(src_mp4)
    if n < 2:
        return (src_mp4, 0, "bad-mp4")
    seed = int.from_bytes(
        hashlib.blake2b(src_mp4.encode("utf-8"), digest_size=4).digest(),
        "little",
    )
    rng = np.random.default_rng(seed=seed)
    n_buttons = len(ACTION_COLS) - 2
    df = {"frame_idx": np.arange(n, dtype=np.int64)}
    btn = rng.integers(0, 2, size=(n, n_buttons), dtype=np.int8).astype(bool)
    for i, col in enumerate(ACTION_COLS[:n_buttons]):
        df[col] = btn[:, i]
    mou = rng.normal(0, 1.5, size=(n, 2)).astype(np.float32)
    df["delta_pitch"] = mou[:, 0]
    df["delta_yaw"]   = mou[:, 1]
    Path(dst_parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(df)),
                   dst_parquet, compression="snappy")
    return (src_mp4, n, "ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-root", required=True, type=Path)
    ap.add_argument("--out-root",   required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    clips_root = args.clips_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "videos").mkdir(parents=True, exist_ok=True)
    (out_root / "actions").mkdir(parents=True, exist_ok=True)

    print(f"Discovering clips under {clips_root} ...")
    clips = sorted(clips_root.rglob("clip_*.mp4"))
    print(f"  {len(clips)} clips")

    jobs = []
    rows = []
    for src in clips:
        rel = src.relative_to(clips_root)
        parts = rel.parts  # demo_X / player_Y / clip_NNN.mp4
        if len(parts) != 3:
            continue
        demo_dir, player_dir, mp4_name = parts
        demo_id = demo_dir.replace("demo_", "")
        player_id = player_dir.replace("player_", "")
        clip_id = int(Path(mp4_name).stem.replace("clip_", ""))

        v_rel = f"videos/demo_{demo_id}/player_{player_id}/{mp4_name}"
        a_rel = f"actions/demo_{demo_id}/player_{player_id}/clip_{clip_id:03d}.parquet"
        v_dst = out_root / v_rel
        a_dst = out_root / a_rel
        jobs.append((str(src), str(v_dst), str(a_dst)))
        rows.append({
            "source_demo_id": demo_id,
            "match_id":       demo_id,
            "player_id":      player_id,
            "clip_id":        clip_id,
            "video_path":     v_rel,
            "actions_path":   a_rel,
            "split":          "train",
            "tick_rate":      64,
            "start_tick":     0,
            "end_tick":       0,
            "fps":            30.0,
        })

    print(f"Synthesising actions parquets in {args.workers} workers ...")
    n_ok = n_skip = n_bad = 0
    frame_lookup: dict[str, int] = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for src_mp4, n, status in ex.map(synth_one, jobs, chunksize=8):
            frame_lookup[src_mp4] = n
            if status == "skip":
                n_skip += 1
            elif status == "ok":
                n_ok += 1
            else:
                n_bad += 1
    print(f"  ok={n_ok}, skip={n_skip}, bad={n_bad}")

    final_rows = []
    for r in rows:
        # Resolve frame count using the source path, since v_dst may be a symlink
        # or copy; we can also re-probe.
        src_mp4_path = jobs[len(final_rows)][0] if len(final_rows) < len(jobs) else None
        frames = frame_lookup.get(src_mp4_path, 0)
        if frames < 2:
            continue
        # End_tick implied; we set frames directly since CSDataset uses 'frames'.
        r["frames"]    = frames
        r["end_tick"]  = int(frames * 64 / 30.0)
        final_rows.append(r)

    print(f"Final manifest: {len(final_rows)} clips, "
          f"{sum(r['frames'] for r in final_rows)} total frames "
          f"(~{sum(r['frames'] for r in final_rows)/30/3600:.2f} h)")

    (out_root / "manifest.json").write_text(json.dumps(final_rows, indent=2))
    print(f"Wrote {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
