#!/usr/bin/env python3
"""Create deterministic CounterStrike-1K release-candidate split manifests.

This script consumes the private HLTV acquisition manifest, estimates each
demo's future POV-video duration, and writes:

* an anonymized candidate-order manifest for release/schema review,
* anonymized cumulative split manifests for training,
* a private render-order manifest that preserves local paths and HLTV retry
  metadata for the rendering/seeding pipeline.

The shuffle is intentionally independent of Python's ``random`` module: rows are
ordered by SHA256(seed || demo_id), which is stable across Python versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "counterstrike-1k.rc_split.v0.1"
DEFAULT_SEED = "counterstrike-1k-dataset:rc1:2026-04-30"
DEFAULT_TARGET_HOURS = (1.0, 10.0, 50.0, 100.0, 500.0, 1000.0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    acquisition_text = args.acquisition_manifest.read_text(encoding="utf-8")
    acquisition_sha256 = _sha256_text(acquisition_text)
    acquisition = json.loads(acquisition_text)
    demos = [dict(row) for row in acquisition.get("demos") or []]
    if not demos:
        raise SystemExit(f"{args.acquisition_manifest} contains no demos")

    stats_by_manifest_id = _load_stats(args.stats_json)
    duration_model = _duration_model(demos, stats_by_manifest_id)
    candidates = [_candidate_row(row, duration_model, args.seed) for row in demos]
    candidates.sort(key=lambda row: (row["shuffle_key"], row["demo_id"]))

    val_rows = _take_until(candidates, target_seconds=args.val_hours * 3600.0)
    val_ids = {row["demo_id"] for row in val_rows}
    train_pool = [row for row in candidates if row["demo_id"] not in val_ids]

    split_targets = _parse_targets(args.train_hours)
    train_splits = {
        _split_name(target): _take_until(train_pool, target_seconds=target * 3600.0)
        for target in split_targets
    }

    memberships: dict[str, list[str]] = {}
    for row in val_rows:
        memberships.setdefault(row["demo_id"], []).append("val")
    for split_name, rows in train_splits.items():
        for row in rows:
            memberships.setdefault(row["demo_id"], []).append(split_name)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output_dir = args.output_dir
    splits_dir = output_dir / "splits"
    private_dir = output_dir / "private"
    splits_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    public_candidates = [
        _public_row(index, row, memberships.get(row["demo_id"], []))
        for index, row in enumerate(candidates)
    ]
    render_rows = [
        _render_row(index, row, memberships.get(row["demo_id"], []))
        for index, row in enumerate(_render_order(val_rows, train_pool))
    ]

    source_hashes = {
        "acquisition_manifest_sha256": acquisition_sha256,
        "stats_json_sha256_by_filename": {
            path.name: _sha256_text(path.read_text(encoding="utf-8"))
            for path in args.stats_json
        },
    }
    private_sources = {
        "acquisition_manifest": str(args.acquisition_manifest),
        "stats_json": [str(path) for path in args.stats_json],
    }

    base_metadata = {
        "dataset_name": args.dataset_name,
        "split_version": args.split_version,
        "schema_version": SCHEMA_VERSION,
        "generated_at": now,
        "seed": args.seed,
        "shuffle_method": "sort by sha256(seed + '\\0' + demo_id), tie-break demo_id",
        "duration_unit": "pov_video_seconds",
        "privacy": {
            "public_split_manifests_contain_hltv_urls": False,
            "public_split_manifests_contain_raw_demo_locations": False,
            "public_split_manifests_contain_steam_ids": False,
            "private_render_order_contains_hltv_retry_metadata": True,
        },
        "source_hashes": source_hashes,
        "policy": {
            "map_scope": "Dust2",
            "demo_scope": "downloaded ok full-demo rows only; multipart and failed rows are excluded by acquisition manifest",
            "validation_policy": "validation is a deterministic prefix of the shuffled candidate order",
            "train_policy": "train splits are cumulative prefixes after removing validation rows",
            "render_failure_policy": (
                "freeze this candidate order, record failed renders separately, then materialize final "
                "splits from render_ok demos using skip-failed-fill-forward semantics"
            ),
        },
        "duration_estimation": duration_model["summary"],
    }

    candidate_manifest = {
        **base_metadata,
        "stats": _candidate_stats(
            candidates,
            val_rows,
            train_pool,
            train_splits,
            val_target_hours=args.val_hours,
        ),
        "candidates": public_candidates,
    }
    _write_json(output_dir / "candidate_order.json", candidate_manifest)

    _write_json(
        private_dir / "render_order_private.json",
        {
            **base_metadata,
            "warning": "Private: contains local demo paths and HLTV URLs. Do not publish.",
            "private_sources": private_sources,
            "stats": _candidate_stats(
                candidates,
                val_rows,
                train_pool,
                train_splits,
                val_target_hours=args.val_hours,
            ),
            "render_order": render_rows,
        },
    )

    split_files: dict[str, str] = {}
    val_manifest = _split_manifest(
        base_metadata=base_metadata,
        split_name="val",
        rows=val_rows,
        target_hours=args.val_hours,
        cumulative_from=None,
    )
    val_path = splits_dir / "val.json"
    _write_json(val_path, val_manifest)
    split_files["val"] = str(val_path)

    for split_name, rows in train_splits.items():
        target = _target_from_name(split_name)
        split_path = splits_dir / f"{split_name}.json"
        _write_json(
            split_path,
            _split_manifest(
                base_metadata=base_metadata,
                split_name=split_name,
                rows=rows,
                target_hours=target,
                cumulative_from="train_pool_after_val",
            ),
        )
        split_files[split_name] = str(split_path)

    index = {
        **base_metadata,
        "stats": _candidate_stats(
            candidates,
            val_rows,
            train_pool,
            train_splits,
            val_target_hours=args.val_hours,
        ),
        "files": {
            "candidate_order": str(output_dir / "candidate_order.json"),
            "private_render_order": str(private_dir / "render_order_private.json"),
            "splits": split_files,
        },
    }
    _write_json(output_dir / "manifest_index.json", index)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_order": str(output_dir / "candidate_order.json"),
                "private_render_order": str(private_dir / "render_order_private.json"),
                "eligible_demos": len(candidates),
                "val": _brief_split_stats(val_rows, args.val_hours),
                "train_splits": {
                    name: _brief_split_stats(rows, _target_from_name(name))
                    for name, rows in train_splits.items()
                },
                "duration_estimation": duration_model["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, action="append", default=[])
    parser.add_argument("--dataset-name", default="counterstrike-1k-dataset")
    parser.add_argument("--split-version", default="rc1")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--val-hours",
        type=float,
        default=10.0,
        help="Validation target in POV-video hours.",
    )
    parser.add_argument(
        "--train-hours",
        default=",".join(_format_hours(target) for target in DEFAULT_TARGET_HOURS),
        help="Comma-separated cumulative train targets in POV-video hours.",
    )
    return parser


def _load_stats(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    by_manifest_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("by_match") or []:
            manifest_id = str(row.get("manifest_id") or "")
            if not manifest_id:
                continue
            copy = dict(row)
            copy["stats_source"] = str(path)
            by_manifest_id[manifest_id] = copy
    return by_manifest_id


def _duration_model(
    demos: list[Mapping[str, Any]],
    stats_by_manifest_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    exact_rows: dict[str, Mapping[str, Any]] = {}
    exact_duration_s = 0.0
    exact_rounds = 0
    exact_clips = 0
    zero_or_empty_stats = 0

    for row in demos:
        manifest_id = str(row.get("manifest_id") or "")
        stats = stats_by_manifest_id.get(manifest_id)
        if not stats:
            continue
        duration_s = float(stats.get("total_duration_s") or 0.0)
        clip_count = int(stats.get("clip_count") or 0)
        round_total = _round_total(row)
        if duration_s <= 0.0 or clip_count <= 0 or round_total is None:
            zero_or_empty_stats += 1
            continue
        exact_rows[manifest_id] = stats
        exact_duration_s += duration_s
        exact_rounds += round_total
        exact_clips += clip_count

    if exact_rounds <= 0:
        round_totals = [_round_total(row) for row in demos if _round_total(row) is not None]
        if not round_totals:
            raise SystemExit("cannot estimate durations: no exact stats and no round totals")
        fallback_seconds_per_round = 689.0
        seconds_per_round = fallback_seconds_per_round
        model_source = "fallback_constant_689_pov_video_seconds_per_round"
    else:
        seconds_per_round = exact_duration_s / exact_rounds
        model_source = "positive_stats_total_duration_div_round_total"

    return {
        "exact_rows": exact_rows,
        "seconds_per_round": seconds_per_round,
        "summary": {
            "method": model_source,
            "exact_positive_stats_rows": len(exact_rows),
            "zero_or_empty_stats_rows_ignored": zero_or_empty_stats,
            "exact_positive_stats_rounds": exact_rounds,
            "exact_positive_stats_clips": exact_clips,
            "exact_positive_stats_pov_video_hours": round(exact_duration_s / 3600.0, 4),
            "estimated_pov_video_seconds_per_round": round(seconds_per_round, 3),
            "estimated_scene_seconds_per_round": round(seconds_per_round / 10.0, 3),
        },
    }


def _candidate_row(
    acquisition_row: Mapping[str, Any],
    duration_model: Mapping[str, Any],
    seed: str,
) -> dict[str, Any]:
    demo_id = str(acquisition_row.get("demo_id") or acquisition_row.get("sha256") or "")
    if not demo_id:
        raise ValueError(f"missing demo_id for {acquisition_row.get('manifest_id')}")
    manifest_id = str(acquisition_row.get("manifest_id") or "")
    exact = duration_model["exact_rows"].get(manifest_id)
    round_total = _round_total(acquisition_row)

    if exact:
        estimated_seconds = float(exact.get("total_duration_s") or 0.0)
        estimate = {
            "method": "exact_parsed_clip_segments",
            "stats_source": exact.get("stats_source"),
            "clip_count": int(exact.get("clip_count") or 0),
            "frame_count": int(exact.get("total_frames") or 0),
        }
    elif round_total is not None:
        estimated_seconds = float(duration_model["seconds_per_round"]) * round_total
        estimate = {
            "method": "round_total_x_estimated_pov_video_seconds_per_round",
            "estimated_pov_video_seconds_per_round": round(
                float(duration_model["seconds_per_round"]), 3
            ),
        }
    else:
        estimated_seconds = 0.0
        estimate = {"method": "missing_round_total_zero_estimate"}

    return {
        "demo_id": demo_id,
        "sha256": str(acquisition_row.get("sha256") or demo_id),
        "manifest_id": manifest_id,
        "match_id": acquisition_row.get("match_id"),
        "entry_index": acquisition_row.get("entry_index"),
        "map": acquisition_row.get("map"),
        "date": acquisition_row.get("date"),
        "event": acquisition_row.get("event"),
        "teams": acquisition_row.get("teams"),
        "round_total": round_total,
        "estimated_pov_video_seconds": round(estimated_seconds, 3),
        "estimated_scene_seconds": round(estimated_seconds / 10.0, 3),
        "estimate": estimate,
        "shuffle_key": _shuffle_key(seed, demo_id),
        "acquisition": dict(acquisition_row),
    }


def _take_until(rows: list[Mapping[str, Any]], *, target_seconds: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0.0
    for row in rows:
        if total >= target_seconds:
            break
        selected.append(dict(row))
        total += float(row.get("estimated_pov_video_seconds") or 0.0)
    return selected


def _render_order(
    val_rows: list[Mapping[str, Any]],
    train_pool: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    ordered: list[Mapping[str, Any]] = []
    for row in [*val_rows, *train_pool]:
        demo_id = str(row.get("demo_id"))
        if demo_id in seen:
            continue
        seen.add(demo_id)
        ordered.append(row)
    return ordered


def _public_row(index: int, row: Mapping[str, Any], memberships: list[str]) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "demo_id": row["demo_id"],
        "map": row.get("map"),
        "round_total": row.get("round_total"),
        "estimated_pov_video_seconds": row.get("estimated_pov_video_seconds"),
        "estimated_scene_seconds": row.get("estimated_scene_seconds"),
        "estimate": row.get("estimate"),
        "shuffle_key": row.get("shuffle_key"),
        "split_memberships": memberships,
    }


def _render_row(index: int, row: Mapping[str, Any], memberships: list[str]) -> dict[str, Any]:
    acquisition = dict(row.get("acquisition") or {})
    return {
        "render_order_index": index,
        "demo_id": row["demo_id"],
        "manifest_id": row.get("manifest_id"),
        "entry_index": row.get("entry_index"),
        "map": row.get("map"),
        "date": row.get("date"),
        "event": row.get("event"),
        "teams": row.get("teams"),
        "round_total": row.get("round_total"),
        "estimated_pov_video_seconds": row.get("estimated_pov_video_seconds"),
        "estimated_scene_seconds": row.get("estimated_scene_seconds"),
        "estimate": row.get("estimate"),
        "shuffle_key": row.get("shuffle_key"),
        "split_memberships": memberships,
        "demo_path": acquisition.get("demo_path"),
        "archive_path": acquisition.get("archive_path"),
        "archive_member": acquisition.get("archive_member"),
        "match_url": acquisition.get("match_url"),
        "resolved_url": acquisition.get("resolved_url"),
        "hltv_demo_id": acquisition.get("hltv_demo_id"),
        "download": acquisition.get("download"),
        "acquisition": acquisition,
    }


def _split_manifest(
    *,
    base_metadata: Mapping[str, Any],
    split_name: str,
    rows: list[Mapping[str, Any]],
    target_hours: float,
    cumulative_from: str | None,
) -> dict[str, Any]:
    return {
        **base_metadata,
        "split_name": split_name,
        "target_pov_video_hours": target_hours,
        "actual_pov_video_hours": round(_total_seconds(rows) / 3600.0, 4),
        "actual_scene_hours": round(_total_seconds(rows) / 36000.0, 4),
        "demo_count": len(rows),
        "complete": _total_seconds(rows) >= target_hours * 3600.0,
        "cumulative_from": cumulative_from,
        "demos": [_public_row(index, row, [split_name]) for index, row in enumerate(rows)],
    }


def _candidate_stats(
    candidates: list[Mapping[str, Any]],
    val_rows: list[Mapping[str, Any]],
    train_pool: list[Mapping[str, Any]],
    train_splits: Mapping[str, list[Mapping[str, Any]]],
    *,
    val_target_hours: float,
) -> dict[str, Any]:
    return {
        "eligible_demos": len(candidates),
        "eligible_pov_video_hours_estimated": round(_total_seconds(candidates) / 3600.0, 4),
        "eligible_scene_hours_estimated": round(_total_seconds(candidates) / 36000.0, 4),
        "val": _brief_split_stats(val_rows, val_target_hours),
        "train_pool": {
            "demo_count": len(train_pool),
            "estimated_pov_video_hours": round(_total_seconds(train_pool) / 3600.0, 4),
            "estimated_scene_hours": round(_total_seconds(train_pool) / 36000.0, 4),
        },
        "train_splits": {
            name: _brief_split_stats(rows, _target_from_name(name))
            for name, rows in train_splits.items()
        },
        "estimate_methods": _count_by(
            str((row.get("estimate") or {}).get("method") or "unknown") for row in candidates
        ),
    }


def _brief_split_stats(rows: list[Mapping[str, Any]], target_hours: float) -> dict[str, Any]:
    total_seconds = _total_seconds(rows)
    return {
        "target_pov_video_hours": target_hours,
        "actual_pov_video_hours": round(total_seconds / 3600.0, 4),
        "actual_scene_hours": round(total_seconds / 36000.0, 4),
        "demo_count": len(rows),
        "complete": total_seconds >= target_hours * 3600.0,
    }


def _round_total(row: Mapping[str, Any]) -> int | None:
    quality = row.get("quality") if isinstance(row.get("quality"), Mapping) else {}
    value = quality.get("round_total")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _parse_targets(value: str) -> list[float]:
    targets = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not targets:
        raise ValueError("--train-hours must contain at least one target")
    return targets


def _split_name(target_hours: float) -> str:
    return f"train_{_format_hours(target_hours)}h"


def _target_from_name(split_name: str) -> float:
    return float(split_name.removeprefix("train_").removesuffix("h"))


def _format_hours(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _total_seconds(rows: Iterable[Mapping[str, Any]]) -> float:
    return sum(float(row.get("estimated_pov_video_seconds") or 0.0) for row in rows)


def _count_by(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _shuffle_key(seed: str, demo_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{demo_id}".encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
