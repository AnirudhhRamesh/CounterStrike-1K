#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.40"]
# ///
"""Freeze a fixed-hours subset of a dataset manifest for baseline training.

Pulls the post-processor's manifest.json from <dataset>'s actions bucket,
runs the same balanced-subset selection logic as `cs2_train/src/download.py`,
and writes a frozen manifest to data/manifests/manifest_{hours}h.json.

The result is the canonical "100h baseline" dataset card: deterministic given
the same `--seed`, sorted, ready to feed into training without any further
sampling.

Match-level train/val/test split:
  - Either `--val-matches N` (random hold-out of N matches), or
  - `--holdout-map NAME` (all clips for that map become the test set).

Usage:
    uv run cs2_train/scripts/freeze_manifest.py \
        --dataset    counterstrike-1k-dataset \
        --hours      100 \
        --out        data/manifests/manifest_100h.json \
        --val-matches 2 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import boto3

# Reuse download.py's selection logic
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_train.src.download import (   # noqa: E402
    assign_splits,
    clip_frames,
    normalize_map_name,
    select_balanced_train_subset,
)


def fetch_remote_manifest(s3, dataset: str) -> list[dict]:
    actions_bucket = f"cs2-{dataset}-actions-s3"
    print(f"Fetching s3://{actions_bucket}/manifest.json ...", flush=True)
    obj = s3.get_object(Bucket=actions_bucket, Key="manifest.json")
    manifest = json.loads(obj["Body"].read())
    print(f"  {len(manifest)} clips in source manifest", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    help="Dataset short name; the actions bucket is "
                         "cs2-<dataset>-actions-s3")
    ap.add_argument("--hours", type=float, required=True,
                    help="Target hours of training data, e.g. 100")
    ap.add_argument("--out", type=Path, required=True,
                    help="Where to write the frozen manifest")
    ap.add_argument("--val-matches", type=int, default=2,
                    help="Number of held-out matches for validation. "
                         "Mutually exclusive with --holdout-map.")
    ap.add_argument("--holdout-map", default=None,
                    help="If set, every match on this map is held out as test")
    ap.add_argument("--holdout-split", default="test",
                    choices=("val", "test"),
                    help="Split label to assign with --holdout-map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--source-manifest", type=Path, default=None,
                    help="Read manifest from a local file instead of S3 "
                         "(for offline / debugging)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Frame rate; only used to convert hours → frames")
    args = ap.parse_args()

    # 1. Source manifest
    if args.source_manifest:
        print(f"Reading {args.source_manifest} ...", flush=True)
        manifest = json.loads(args.source_manifest.read_text())
        print(f"  {len(manifest)} clips", flush=True)
    else:
        session = boto3.Session(region_name=args.region)
        s3 = session.client("s3")
        manifest = fetch_remote_manifest(s3, args.dataset)

    if not manifest:
        sys.exit("Empty manifest — has the post-processor run yet?")

    # 2. Assign match-level splits.
    splits = assign_splits(
        manifest,
        val_matches=args.val_matches,
        seed=args.seed,
        holdout_map=args.holdout_map,
        holdout_split=args.holdout_split,
    )
    for clip in manifest:
        clip["split"] = splits[clip["source_demo_id"]]

    # 3. Pick a balanced training subset of `hours` total.
    target_total_frames = int(round(args.hours * 3600.0 * args.fps))
    train_rows = [c for c in manifest if c["split"] == "train"]
    if not train_rows:
        sys.exit("No train rows after split — check --holdout-map / --val-matches.")
    if any(not c.get("map") for c in train_rows):
        # 1k dataset is single-map (Dust 2). Inject the map field if missing
        # so select_balanced_train_subset's invariant holds.
        for c in train_rows:
            c.setdefault("map", "de_dust2")

    selected_train, scale_stats = select_balanced_train_subset(
        train_rows,
        target_total_frames=target_total_frames,
        seed=args.seed,
        fps=args.fps,
    )

    # 4. Build output manifest = selected train + all val/test rows.
    holdout_rows = [c for c in manifest if c["split"] != "train"]
    out_rows = sorted(
        selected_train + holdout_rows,
        key=lambda c: (c.get("map", ""),
                       str(c.get("source_demo_id", "")),
                       str(c.get("player_id", "")),
                       int(c.get("clip_id", 0))),
    )

    # 5. Stats
    by_split: dict[str, int] = defaultdict(int)
    by_split_frames: dict[str, int] = defaultdict(int)
    for c in out_rows:
        by_split[c["split"]] += 1
        by_split_frames[c["split"]] += clip_frames(c)
    selected_hours = round(scale_stats["selected_total_frames"] / args.fps / 3600.0, 4)

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_rows, indent=2))
    stats_path = args.out.with_suffix(".stats.json")
    stats = {
        "dataset":               args.dataset,
        "target_hours":          args.hours,
        "selected_hours":        selected_hours,
        "n_clips_total":         len(out_rows),
        "n_clips_by_split":      dict(by_split),
        "n_frames_by_split":     dict(by_split_frames),
        "frames_by_map":         scale_stats["frames_by_map"],
        "source_demos_by_map":   scale_stats["source_demos_by_map"],
        "seed":                  args.seed,
        "val_matches":           args.val_matches,
        "holdout_map":           args.holdout_map,
        "holdout_split":         args.holdout_split,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nFrozen manifest:")
    print(f"  target hours:    {args.hours}")
    print(f"  selected hours:  {selected_hours}")
    print(f"  total clips:     {len(out_rows)}")
    print(f"  by split:        {dict(by_split)}")
    print(f"  by split frames: {dict(by_split_frames)}")
    print(f"  -> {args.out}")
    print(f"  -> {stats_path}")


if __name__ == "__main__":
    main()
