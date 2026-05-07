"""Build a CSDataset-compatible manifest for the benchmark subset.

Reads the original manifest pulled from S3, keeps only entries whose
`video_s3` resolves to a clip we actually have on local disk under
<root>/raw/videos/, and writes <root>/raw/manifest.json with the
download.py field aliases (video_path, actions_path, match_id, split).

Usage:
    python prepare_manifest.py \
        --src-manifest /opt/dlami/nvme/bench/manifest_raw.json \
        --root         /opt/dlami/nvme/bench/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def relpath_from_uri(uri: str) -> str:
    """videos/<dataset>/<demo>/<player>/clip_NNN.mp4 -> videos/<demo>/<player>/clip_NNN.mp4"""
    _, key = parse_s3_uri(uri)
    parts = key.split("/")
    if parts[0] in ("videos", "actions") and len(parts) >= 5:
        return "/".join([parts[0]] + parts[2:])
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-manifest", required=True)
    ap.add_argument("--root", required=True, help="Local root (must contain videos/ and actions/)")
    ap.add_argument("--out", default=None, help="Output path (defaults to <root>/manifest.json)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "manifest.json"

    src = json.loads(Path(args.src_manifest).read_text())
    new = []
    n_kept = n_dropped = 0
    for clip in src:
        v_rel = relpath_from_uri(clip["video_s3"])
        a_rel = relpath_from_uri(clip["actions_s3"])
        v_local = root / v_rel
        a_local = root / a_rel
        if not (v_local.exists() and a_local.exists()):
            n_dropped += 1
            continue
        new.append({
            **clip,
            "match_id":     clip["source_demo_id"],
            "video_path":   v_rel,
            "actions_path": a_rel,
            "split":        "train",   # everything is "train" for the benchmark
            "tick_rate":    clip.get("tick_rate", 64),
        })
        n_kept += 1

    out.write_text(json.dumps(new, indent=2))
    total_frames = sum(int(c.get("frames", 0)) for c in new)
    print(f"kept {n_kept} clips, dropped {n_dropped} not on disk")
    print(f"total frames {total_frames}, ~{total_frames / 30 / 60:.1f} min @ 30 FPS")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
