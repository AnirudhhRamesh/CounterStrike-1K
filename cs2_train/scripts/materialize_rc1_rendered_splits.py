#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.40"]
# ///
"""Freeze the rc1 rendered split manifests for `counterstrike-1k-dataset`.

The rc1 candidate order is frozen in
`cs2_train/data/manifests/counterstrike-1k-dataset/rc1/private/render_order_private.json`
and the render-failure policy says:

    "freeze this candidate order, record failed renders separately, then
     materialize final splits from render_ok demos using
     skip-failed-fill-forward semantics"

This script implements that materialization step. For each demo in
candidate order, we look at the renders DDB for state=Succeeded clip rows.
A demo is "render_ok" iff it has at least one Succeeded clip. We walk the
render_order indices ascending and assemble:

  • val           : prefix of render_ok demos until target_pov_video_hours hit
  • train_1h      : next prefix until 1h hit
  • train_10h     : train_1h ∪ next prefix until 10h hit (cumulative)
  • train_50h     : train_10h ∪ next prefix until 50h hit
  • train_100h    : train_50h ∪ next prefix until 100h hit
  …

For each split S we emit ONE manifest at
  cs2_train/data/manifests/<dataset>/<rc>/rendered/<S>.json
that contains every clip belonging to (val ∪ S) with `split` set per row to
"val" or "train" — the shape `cs2_train.src.dataset.CSDataset` already
understands. The manifest also carries `video_s3` / `actions_s3` /
`debug_s3` URIs, plus `video_path` / `actions_path` relative paths matching
what `cs2_train/src/download.py` produces; both views are in sync.

Note: this script does NOT depend on `cs2_generate/post_process.py` having
run yet. It only needs the renders DDB to be populated with Succeeded
rows. The `actions_s3` URIs in the output point to where post_process WILL
place the parquets — once it runs, the manifest is ready for download.py
without any rewrite. Until then the manifest is a frozen artifact (correct
demo+clip identity, S3 layout pinned).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key


# Public clip metadata mirrored verbatim from
# cs2_generate/post_process.py:PUBLIC_CLIP_METADATA_KEYS so the rendered
# manifest fields stay in lockstep with what post_process emits.
PUBLIC_CLIP_METADATA_KEYS = (
    "manifest_id",
    "hltv_match_id",
    "hltv_match_url",
    "map",
    "date",
    "event",
    "teams",
)

# Default split targets in pov-video-hours. These mirror the values frozen
# in `manifest_index.json`'s `stats.train_splits.<S>.target_pov_video_hours`
# and `stats.val.target_pov_video_hours`.
DEFAULT_TARGETS_HOURS = OrderedDict([
    ("val",         20.0),
    ("train_1h",     1.0),
    ("train_10h",   10.0),
    ("train_50h",   50.0),
    ("train_100h", 100.0),
    ("train_500h", 500.0),
    ("train_1000h", 1000.0),
])
TICK_RATE = 64


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    rc_dir = (
        repo_root
        / "cs2_train" / "data" / "manifests"
        / args.dataset / args.rc / "private"
    )
    private_path = rc_dir / "render_order_private.json"
    if not private_path.exists():
        sys.exit(f"render_order_private.json not found at {private_path}")
    private = json.loads(private_path.read_text(encoding="utf-8"))

    # 1. Pull all Succeeded clip rows from DDB.
    ddb = boto3.resource("dynamodb", region_name=args.region)
    clips_table_name = f"cs2-{args.dataset}-clips"
    table = ddb.Table(clips_table_name)
    print(f"Querying DDB {clips_table_name} for run_id={args.run_id} ...", flush=True)
    succeeded = _query_succeeded_by_run(table, args.run_id)
    print(f"  {len(succeeded)} Succeeded clip rows", flush=True)

    # 2. Group by source_demo_id; render_ok = (>= 1 Succeeded clip).
    by_demo: dict[str, list[dict]] = defaultdict(list)
    for clip in succeeded:
        by_demo[str(clip["source_demo_id"])].append(clip)

    # 3. Walk candidate-order, fill val then cumulative-prefix train splits.
    targets_hours = _parse_target_overrides(args.targets, DEFAULT_TARGETS_HOURS)
    targets_seconds = OrderedDict((k, v * 3600.0) for k, v in targets_hours.items())

    candidates = sorted(
        private["render_order"], key=lambda e: int(e["render_order_index"])
    )
    val_demos: list[dict] = []
    train_tiers: OrderedDict[str, list[dict]] = OrderedDict(
        (s, []) for s in targets_seconds if s != "val"
    )
    val_seconds = 0.0
    train_seconds = 0.0
    train_tier_order = list(train_tiers.keys())
    train_tier_idx = 0
    seen_demo_ids: set[str] = set()

    for entry in candidates:
        demo_id_full = str(entry["demo_id"])
        # DDB stores source_demo_id = sha256[:16] (see
        # cs2_hltv/src/cs2_hltv/aws_pipeline.py); render_order_private uses
        # the full sha256. Join on the prefix.
        demo_id_short = demo_id_full[:16]
        if demo_id_short in seen_demo_ids:
            continue
        clips = by_demo.get(demo_id_short, [])
        if not clips:
            continue  # render_failed → skip-and-fill-forward
        seen_demo_ids.add(demo_id_short)
        demo_seconds = sum(_to_float(c.get("duration_s", 0)) for c in clips)
        if val_seconds < targets_seconds["val"]:
            val_demos.append({"entry": entry, "clips": clips,
                              "duration_s": demo_seconds})
            val_seconds += demo_seconds
            continue
        # Walk the train tiers in order, filling each cumulative-prefix
        # until its target is hit. A demo never lands in two tiers.
        if train_tier_idx >= len(train_tier_order):
            break
        cur_tier = train_tier_order[train_tier_idx]
        train_tiers[cur_tier].append(
            {"entry": entry, "clips": clips, "duration_s": demo_seconds}
        )
        train_seconds += demo_seconds
        if train_seconds >= targets_seconds[cur_tier]:
            train_tier_idx += 1

    # 4. Emit per-split manifests.
    rendered_dir = repo_root / "cs2_train" / "data" / "manifests" / args.dataset / args.rc / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    actions_bucket = f"cs2-{args.dataset}-actions-s3"
    debug_bucket = f"cs2-{args.dataset}-debug-s3"

    summaries: list[dict] = []
    cumulative_train: list[dict] = []
    for tier_name, tier_demos in train_tiers.items():
        cumulative_train.extend(tier_demos)
        summary = _write_split_manifest(
            split_name=tier_name,
            val_demos=val_demos,
            train_demos=cumulative_train,
            target_hours=targets_hours[tier_name],
            val_target_hours=targets_hours["val"],
            actions_bucket=actions_bucket,
            debug_bucket=debug_bucket,
            dataset_name=args.dataset,
            rc=args.rc,
            run_id=args.run_id,
            seed=private.get("seed"),
            schema_version=private.get("schema_version"),
            generated_at=private.get("generated_at"),
            out_dir=rendered_dir,
        )
        summaries.append(summary)

    # 5. Emit a val-only manifest too, since a future evaluation run may
    # want it without paying for any train rows.
    summary = _write_split_manifest(
        split_name="val",
        val_demos=val_demos,
        train_demos=[],
        target_hours=targets_hours["val"],
        val_target_hours=targets_hours["val"],
        actions_bucket=actions_bucket,
        debug_bucket=debug_bucket,
        dataset_name=args.dataset,
        rc=args.rc,
        run_id=args.run_id,
        seed=private.get("seed"),
        schema_version=private.get("schema_version"),
        generated_at=private.get("generated_at"),
        out_dir=rendered_dir,
    )
    summaries.insert(0, summary)

    # 6. Index file summarising what we just wrote.
    index = {
        "dataset_name": args.dataset,
        "rc_version": args.rc,
        "run_id": args.run_id,
        "schema_version": "counterstrike-1k.rc_split_rendered.v0.1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "render_order_seed": private.get("seed"),
        "render_order_schema_version": private.get("schema_version"),
        "render_order_generated_at": private.get("generated_at"),
        "splits": summaries,
        "policy": {
            "skip_failed_fill_forward": (
                "Demos without any Succeeded clip in DDB are skipped; the "
                "next render_ok demo in candidate order takes the slot."
            ),
            "split_nesting": (
                "val rows are written into every train_*h manifest with split='val'. "
                "Train tiers are cumulative prefixes: train_1h ⊆ train_10h ⊆ train_50h ⊆ train_100h ⊆ ..."
            ),
        },
    }
    index_path = rendered_dir / "rendered_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(f"\nIndex: {index_path.relative_to(repo_root)}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="counterstrike-1k-dataset")
    p.add_argument("--rc", default="rc1")
    p.add_argument("--run-id", default="counterstrike-1k-dataset",
                   help="run_id used by the render fleet (DDB GSI key).")
    p.add_argument("--region", default="us-east-1")
    p.add_argument(
        "--targets",
        default=None,
        help=(
            "Optional override of split targets. Comma-separated NAME=HOURS, "
            "e.g. 'val=20,train_100h=100'. Splits not listed keep defaults."
        ),
    )
    return p.parse_args()


def _parse_target_overrides(raw: str | None,
                            defaults: OrderedDict[str, float]
                            ) -> OrderedDict[str, float]:
    if not raw:
        return OrderedDict(defaults)
    out = OrderedDict(defaults)
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise SystemExit(f"--targets entry must be NAME=HOURS, got {tok!r}")
        name, val = tok.split("=", 1)
        out[name.strip()] = float(val)
    return out


def _query_succeeded_by_run(table, run_id: str) -> list[dict]:
    items: list[dict] = []
    last = None
    while True:
        kw = dict(
            IndexName="run_id-index",
            KeyConditionExpression=Key("run_id").eq(run_id),
            FilterExpression=boto3.dynamodb.conditions.Attr("state").eq("Succeeded"),
        )
        if last:
            kw["ExclusiveStartKey"] = last
        r = table.query(**kw)
        items.extend(r.get("Items", []))
        last = r.get("LastEvaluatedKey")
        if not last:
            return items


def _write_split_manifest(
    *,
    split_name: str,
    val_demos: list[dict],
    train_demos: list[dict],
    target_hours: float,
    val_target_hours: float,
    actions_bucket: str,
    debug_bucket: str,
    dataset_name: str,
    rc: str,
    run_id: str,
    seed: str | None,
    schema_version: str | None,
    generated_at: str | None,
    out_dir: Path,
) -> dict:
    rows = []
    rows.extend(_emit_rows(val_demos, "val", dataset_name, actions_bucket, debug_bucket))
    rows.extend(_emit_rows(train_demos, "train", dataset_name, actions_bucket, debug_bucket))

    val_seconds = sum(d["duration_s"] for d in val_demos)
    train_seconds = sum(d["duration_s"] for d in train_demos)

    # Per-split file is a flat list — drop-in compatible with
    # `cs2_train.src.dataset.CSDataset(manifest_name=...)` which iterates
    # the JSON top-level and filters by `split`. The full envelope (schema
    # version, stats, provenance) goes into rendered_index.json next to it.
    out_path = out_dir / f"{split_name}.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str) + "\n",
                        encoding="utf-8")
    rel = out_path.relative_to(out_dir.parents[3])
    print(
        f"  {split_name:<12} demos={len(val_demos)+len(train_demos):>3d} "
        f"clips={len(rows):>5d} "
        f"val={val_seconds/3600.0:6.2f}h "
        f"train={train_seconds/3600.0:7.2f}h "
        f"→ {rel}",
        flush=True,
    )
    return {
        "split_name": split_name,
        "path": str(out_path.relative_to(out_dir.parents[3])),
        "clip_count": len(rows),
        "val_actual_hours": round(val_seconds / 3600.0, 4),
        "train_actual_hours": round(train_seconds / 3600.0, 4),
        "val_demo_count": len(val_demos),
        "train_demo_count": len(train_demos),
    }


def _emit_rows(
    demos: list[dict],
    split_label: str,
    dataset_name: str,
    actions_bucket: str,
    debug_bucket: str,
) -> list[dict]:
    rows: list[dict] = []
    for demo in demos:
        entry = demo["entry"]
        clips = sorted(
            demo["clips"],
            key=lambda c: (str(c["player_id"]), int(_to_int(c["clip_id"]))),
        )
        for clip in clips:
            rel = (
                f"demo_{clip['source_demo_id']}/player_{clip['player_id']}"
                f"/clip_{_to_int(clip['clip_id']):03d}"
            )
            video_path = f"videos/{rel}.mp4"
            actions_path = f"actions/{rel}.parquet"
            debug_path = f"debug/{rel}.mp4"
            row = {
                "split": split_label,
                "render_order_index": int(entry["render_order_index"]),
                "source_demo_id": str(clip["source_demo_id"]),
                "match_id": str(clip["source_demo_id"]),  # alias for legacy CSDataset
                "player_id": str(clip["player_id"]),
                "player_name": str(clip.get("player_name", "")),
                "clip_id": _to_int(clip["clip_id"]),
                "start_tick": _to_int(clip["start_tick"]),
                "end_tick": _to_int(clip["end_tick"]),
                "frames": _to_int(clip.get("frame_count", clip.get("frames", 0))),
                "fps": _to_float(clip.get("fps", 30)),
                "tick_rate": TICK_RATE,
                "duration_s": _to_float(clip.get("duration_s", 0)),
                "video_s3": str(clip.get("render_s3_uri", "")),
                "actions_s3": (
                    f"s3://{actions_bucket}/actions/{dataset_name}/{rel}.parquet"
                ),
                "debug_s3": (
                    f"s3://{debug_bucket}/debug/{dataset_name}/{rel}.mp4"
                ),
                "video_path": video_path,
                "actions_path": actions_path,
                "debug_path": debug_path,
            }
            for key in PUBLIC_CLIP_METADATA_KEYS:
                if key in clip and clip[key] not in (None, ""):
                    row[key] = _json_safe(clip[key])
            rows.append(row)
    return rows


def _to_int(v) -> int:
    if isinstance(v, Decimal):
        return int(v)
    return int(v)


def _to_float(v) -> float:
    if isinstance(v, Decimal):
        return float(v)
    return float(v)


def _json_safe(v):
    if isinstance(v, Decimal):
        if v == v.to_integral_value():
            return int(v)
        return float(v)
    if isinstance(v, list):
        return [_json_safe(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    return v


if __name__ == "__main__":
    raise SystemExit(main())
