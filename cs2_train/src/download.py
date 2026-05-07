"""Download a CS2-WM dataset from its three S3 buckets to local disk.

The post-processor (cs2_generate/post_process.py) writes:
  s3://cs2-{DATASET_NAME}-actions-s3/manifest.json
  s3://cs2-{DATASET_NAME}-actions-s3/stats.json
  s3://cs2-{DATASET_NAME}-actions-s3/actions/.../clip_NNN.parquet
  s3://cs2-{DATASET_NAME}-clips-s3/videos/.../clip_NNN.mp4
  s3://cs2-{DATASET_NAME}-debug-s3/debug/.../clip_NNN.mp4

This script pulls everything to a local directory with the layout the
training Dataset expects:
  <dest>/
    manifest.json          (rewritten with relative paths + train/val split)
    stats.json
    videos/<demo>/<player>/clip_NNN.mp4
    actions/<demo>/<player>/clip_NNN.parquet
    debug/<demo>/<player>/clip_NNN.mp4    (only if --include-debug)

Train/val split is at the *match* level — never at sample/clip level —
so the model can't memorise per-round textures.

Usage:
    uv run cs2_train/src/download.py \\
        --dataset cs2-dataset-skypilot-v1-3-demos \\
        --dest /opt/dlami/nvme/cs2-data \\
        --holdout-map Nuke \\
        --scaling-hours 1,10,50,100,200,400,500
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from tqdm import tqdm


def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def relpath_from_video_uri(uri: str) -> str:
    """s3://...-clips-s3/videos/<dataset>/<demo>/<player>/clip_NNN.mp4
    → videos/<demo>/<player>/clip_NNN.mp4 (drop the dataset segment)."""
    _, key = parse_s3_uri(uri)
    parts = key.split("/")
    # key = videos/<dataset>/<demo>/<player>/<clip>
    if parts[0] in ("videos", "actions", "debug") and len(parts) >= 5:
        return "/".join([parts[0]] + parts[2:])
    return key  # fallback: keep as-is


def normalize_map_name(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower().strip()
    text = re.sub(r"^de[_\-\s]+", "", text)
    text = text.replace("dust_2", "dust2")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def clip_frames(clip: dict) -> int:
    n_frames = int(clip.get("frames", 0))
    if n_frames > 0:
        return n_frames
    tick_rate = int(clip.get("tick_rate", 64))
    fps = float(clip.get("fps", 30))
    ticks = int(clip["end_tick"]) - int(clip["start_tick"])
    return int(round(ticks / tick_rate * fps))


def assign_splits(
    manifest: list[dict],
    val_matches: int,
    seed: int,
    *,
    holdout_map: str | None = None,
    holdout_split: str = "test",
) -> dict[str, str]:
    """Match-level split. Returns {source_demo_id: split}."""
    matches = sorted({c["source_demo_id"] for c in manifest})
    if holdout_map:
        target_map = normalize_map_name(holdout_map)
        if not target_map:
            raise ValueError("--holdout-map cannot be empty")

        splits: dict[str, str] = {}
        held_out = 0
        missing_map_metadata: list[str] = []
        by_match: dict[str, list[dict]] = defaultdict(list)
        for clip in manifest:
            by_match[clip["source_demo_id"]].append(clip)
        for match_id in matches:
            maps = {
                normalize_map_name(clip.get("map"))
                for clip in by_match[match_id]
                if clip.get("map")
            }
            if not maps:
                missing_map_metadata.append(match_id)
                continue
            split = holdout_split if target_map in maps else "train"
            splits[match_id] = split
            if split == holdout_split:
                held_out += 1

        if missing_map_metadata:
            preview = ", ".join(missing_map_metadata[:5])
            raise ValueError(
                "--holdout-map requires map metadata in manifest rows; "
                f"missing for {len(missing_map_metadata)} match(es), e.g. {preview}"
            )
        if held_out == 0:
            raise ValueError(f"--holdout-map {holdout_map!r} did not match any manifest rows")
        return splits

    rng = random.Random(seed)
    rng.shuffle(matches)
    val_set = set(matches[:val_matches])
    return {m: ("val" if m in val_set else "train") for m in matches}


def summarize_frames_by(rows: list[dict], key: str) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(key)
        if value:
            totals[str(value)] += clip_frames(row)
    return dict(sorted(totals.items()))


def parse_hours_list(value: str | None) -> list[float]:
    if not value:
        return []
    hours: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = float(part)
        if parsed <= 0:
            raise ValueError("--scaling-hours values must be positive")
        hours.append(parsed)
    return hours


def write_scaling_manifests(
    manifest: list[dict],
    *,
    dest: Path,
    hours: list[float],
    fps: float,
    seed: int,
    manifest_dir: Path,
) -> list[Path]:
    if not hours:
        return []

    train_rows = [row for row in manifest if row.get("split") == "train"]
    heldout_rows = [row for row in manifest if row.get("split") != "train"]
    if not train_rows:
        raise ValueError("cannot write scaling manifests without train rows")
    if any(not row.get("map") for row in train_rows):
        raise ValueError("scaling manifests require map metadata on train rows")

    output_dir = dest / manifest_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for target_hours in hours:
        target_total_frames = int(round(target_hours * 3600.0 * fps))
        selected_train, scale_stats = select_balanced_train_subset(
            train_rows,
            target_total_frames=target_total_frames,
            seed=seed,
            fps=fps,
        )
        subset = selected_train + heldout_rows
        label = hours_label(target_hours)
        manifest_path = output_dir / f"manifest_train_{label}.json"
        stats_path = output_dir / f"manifest_train_{label}.stats.json"
        manifest_path.write_text(json.dumps(subset, indent=2))
        stats_path.write_text(json.dumps(scale_stats, indent=2, sort_keys=True))
        written.append(manifest_path)
    return written


def select_balanced_train_subset(
    rows: list[dict],
    *,
    target_total_frames: int,
    seed: int,
    fps: float,
) -> tuple[list[dict], dict]:
    units_by_map: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_demo_id"])].append(row)

    for source_demo_id, unit_rows in grouped.items():
        maps = {str(row["map"]) for row in unit_rows if row.get("map")}
        if len(maps) != 1:
            raise ValueError(f"source_demo_id {source_demo_id} has inconsistent map metadata: {sorted(maps)}")
        map_name = next(iter(maps))
        units_by_map[map_name].append(
            {
                "source_demo_id": source_demo_id,
                "frames": sum(clip_frames(row) for row in unit_rows),
                "rows": unit_rows,
            }
        )

    maps = sorted(units_by_map)
    target_frames_per_map = max(1, int(round(target_total_frames / len(maps))))
    selected_rows: list[dict] = []
    frames_by_map: dict[str, int] = {}
    units_by_map_count: dict[str, int] = {}
    for map_index, map_name in enumerate(maps):
        rng = random.Random(seed + map_index)
        picked = pick_units_to_target(
            units_by_map[map_name],
            target_frames=target_frames_per_map,
            rng=rng,
        )
        for unit in picked:
            selected_rows.extend(unit["rows"])
        frames_by_map[map_name] = sum(int(unit["frames"]) for unit in picked)
        units_by_map_count[map_name] = len(picked)

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (row.get("map", ""), row.get("source_demo_id", ""), row.get("player_id", ""), row.get("clip_id", 0)),
    )
    stats = {
        "target_total_frames": target_total_frames,
        "target_frames_per_map": target_frames_per_map,
        "selected_total_frames": sum(frames_by_map.values()),
        "selected_hours": round(sum(frames_by_map.values()) / fps / 3600.0, 4),
        "frames_by_map": dict(sorted(frames_by_map.items())),
        "source_demos_by_map": dict(sorted(units_by_map_count.items())),
        "train_clips": len(selected_rows),
    }
    return selected_rows, stats


def pick_units_to_target(units: list[dict], *, target_frames: int, rng: random.Random) -> list[dict]:
    candidates = list(units)
    rng.shuffle(candidates)
    selected: list[dict] = []
    selected_frames = 0
    while candidates and selected_frames < target_frames:
        remaining = target_frames - selected_frames
        under = [unit for unit in candidates if int(unit["frames"]) <= remaining]
        if under:
            pick = max(under, key=lambda unit: (int(unit["frames"]), str(unit["source_demo_id"])))
        else:
            pick = min(candidates, key=lambda unit: (int(unit["frames"]), str(unit["source_demo_id"])))
        selected.append(pick)
        selected_frames += int(pick["frames"])
        candidates.remove(pick)
    return selected


def hours_label(hours: float) -> str:
    if hours.is_integer():
        return f"{int(hours):03d}h"
    return f"{hours:g}h".replace(".", "p")


def download_one(args: tuple) -> tuple[str, bool, str]:
    """Download one (bucket, key, dest). Returns (key, ok, err)."""
    s3, bucket, key, dest = args
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return (key, True, "skip")
    try:
        s3.download_file(bucket, key, str(dest))
        return (key, True, "ok")
    except Exception as e:  # noqa: BLE001
        return (key, False, repr(e))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="DATASET_NAME (e.g. cs2-dataset-skypilot-v1-3-demos)")
    ap.add_argument("--dest", required=True, help="Local directory to write to")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--val-matches", type=int, default=1, help="Number of matches to hold out for validation")
    ap.add_argument(
        "--holdout-map",
        help="Hold out one map name for out-of-distribution evaluation; all other maps become train.",
    )
    ap.add_argument(
        "--holdout-split",
        choices=("val", "test"),
        default="test",
        help="Split label to use with --holdout-map.",
    )
    ap.add_argument("--include-debug", action="store_true", help="Also download debug overlay videos")
    ap.add_argument("--workers", type=int, default=16, help="Concurrent S3 downloads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="Only download first N clips (debug)")
    ap.add_argument(
        "--scaling-hours",
        help="Comma-separated train subset sizes in rendered hours, e.g. 1,10,50,100,200,400,500.",
    )
    ap.add_argument(
        "--scaling-manifest-dir",
        type=Path,
        default=Path("manifests"),
        help="Directory under --dest for generated scaling manifests.",
    )
    ap.add_argument("--manifest-fps", type=float, default=30.0, help="FPS used to convert scaling hours to frames")
    args = ap.parse_args()

    actions_bucket = f"cs2-{args.dataset}-actions-s3"
    clips_bucket = f"cs2-{args.dataset}-clips-s3"
    debug_bucket = f"cs2-{args.dataset}-debug-s3"

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    cfg = Config(region_name=args.region, max_pool_connections=max(50, args.workers * 2))
    session = boto3.Session(region_name=args.region)
    s3 = session.client("s3", config=cfg)

    # 1. Fetch manifest + stats
    print(f"Fetching manifest from s3://{actions_bucket}/manifest.json")
    s3.download_file(actions_bucket, "manifest.json", str(dest / "manifest.json"))
    s3.download_file(actions_bucket, "stats.json", str(dest / "stats.json"))
    manifest = json.loads((dest / "manifest.json").read_text())
    stats = json.loads((dest / "stats.json").read_text())
    print(f"  {stats['total_clips']} clips, {stats['total_matches']} matches, {stats['total_duration_s']/3600:.2f}h")

    if args.limit:
        manifest = manifest[: args.limit]
        print(f"  --limit {args.limit}: truncated to {len(manifest)} clips")

    # 2. Match-level split
    splits = assign_splits(
        manifest,
        val_matches=args.val_matches,
        seed=args.seed,
        holdout_map=args.holdout_map,
        holdout_split=args.holdout_split,
    )
    split_counts = {split: sum(1 for value in splits.values() if value == split) for split in sorted(set(splits.values()))}
    if args.holdout_map:
        heldout_ids = sorted([m for m, s in splits.items() if s == args.holdout_split])
        print(
            f"Split: map holdout {args.holdout_map!r} -> "
            f"{split_counts.get('train', 0)} train + {len(heldout_ids)} {args.holdout_split} match(es)"
        )
    else:
        val_match_ids = sorted([m for m, s in splits.items() if s == "val"])
        print(f"Split: {sum(1 for s in splits.values() if s == 'train')} train + "
              f"{len(val_match_ids)} val match(es): {val_match_ids}")

    # 3. Build download jobs + rewrite manifest with relative paths
    jobs = []
    new_manifest = []
    for clip in manifest:
        v_bucket, v_key = parse_s3_uri(clip["video_s3"])
        a_bucket, a_key = parse_s3_uri(clip["actions_s3"])
        v_rel = relpath_from_video_uri(clip["video_s3"])
        a_rel = relpath_from_video_uri(clip["actions_s3"])
        jobs.append((s3, v_bucket, v_key, dest / v_rel))
        jobs.append((s3, a_bucket, a_key, dest / a_rel))
        new_clip = {
            **clip,
            "match_id": clip["source_demo_id"],   # alias for older dataset code
            "video_path": v_rel,
            "actions_path": a_rel,
            "split": splits[clip["source_demo_id"]],
            "tick_rate": clip.get("tick_rate", 64),
        }
        if args.include_debug:
            d_bucket, d_key = parse_s3_uri(clip["debug_s3"])
            d_rel = relpath_from_video_uri(clip["debug_s3"])
            jobs.append((s3, d_bucket, d_key, dest / d_rel))
            new_clip["debug_path"] = d_rel
        new_manifest.append(new_clip)

    # 4. Pull in parallel
    n_ok = n_skip = n_err = 0
    errors: list[tuple[str, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, j) for j in jobs]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="download"):
            key, ok, msg = fut.result()
            if not ok:
                n_err += 1
                errors.append((key, msg))
            elif msg == "skip":
                n_skip += 1
            else:
                n_ok += 1

    print(f"  downloaded: {n_ok}, skipped (already present): {n_skip}, errors: {n_err}")
    if errors:
        for k, e in errors[:5]:
            print(f"  ERR {k}: {e}")

    # 5. Write the local manifest
    out_manifest = dest / "manifest.json"
    out_manifest.write_text(json.dumps(new_manifest, indent=2))
    local_stats = {
        **stats,
        "local_manifest": {
            "clips": len(new_manifest),
            "splits": {
                split: sum(1 for row in new_manifest if row["split"] == split)
                for split in sorted({row["split"] for row in new_manifest})
            },
            "frames_by_map": summarize_frames_by(new_manifest, "map"),
            "train_frames_by_map": summarize_frames_by(
                [row for row in new_manifest if row["split"] == "train"],
                "map",
            ),
            "split_policy": (
                {"holdout_map": args.holdout_map, "holdout_split": args.holdout_split}
                if args.holdout_map
                else {"val_matches": args.val_matches, "seed": args.seed}
            ),
        },
    }
    (dest / "stats.json").write_text(json.dumps(local_stats, indent=2, sort_keys=True, default=str))

    scaling_hours = parse_hours_list(args.scaling_hours)
    scaling_manifests = write_scaling_manifests(
        new_manifest,
        dest=dest,
        hours=scaling_hours,
        fps=args.manifest_fps,
        seed=args.seed,
        manifest_dir=args.scaling_manifest_dir,
    )
    print(f"Wrote {out_manifest} ({len(new_manifest)} clips)")
    if local_stats["local_manifest"]["frames_by_map"]:
        print(f"Frames by map: {local_stats['local_manifest']['frames_by_map']}")
    if scaling_manifests:
        print("Scaling manifests:")
        for path in scaling_manifests:
            print(f"  {path.relative_to(dest)}")
    print(f"Done: {dest}")


if __name__ == "__main__":
    main()
