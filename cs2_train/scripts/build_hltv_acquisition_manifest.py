#!/usr/bin/env python
"""Build the private HLTV acquisition manifest used before public split creation.

This joins the curated HLTV source manifest with one or more downloader status
JSONL files, deduplicates retries by manifest entry, and filters rows that
should not enter CounterStrike-1K v1. In particular, archive members named
``*-p1.dem``, ``*-p2.dem``, etc. are treated as partial map recordings and are
excluded from the full-demo acquisition manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HLTV_SRC = ROOT / "cs2_hltv" / "src"
if str(HLTV_SRC) not in sys.path:
    sys.path.insert(0, str(HLTV_SRC))

from cs2_hltv.archive import split_demo_part_number  # noqa: E402
from cs2_hltv.manifest import assert_no_steam_ids, load_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source_text = args.source_manifest.read_text(encoding="utf-8")
    assert_no_steam_ids(source_text, source=str(args.source_manifest))
    source_payload = json.loads(source_text)
    source_rows = _manifest_rows(source_payload)
    source_entries = load_manifest(args.source_manifest)
    source_by_index = {
        entry.index: (entry, dict(source_rows[entry.index])) for entry in source_entries
    }

    status_by_index, status_count = _load_status_rows(args.status_jsonl)
    demos: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()

    for entry in source_entries:
        status_row = status_by_index.get(entry.index)
        reject_reason = _reject_reason(status_row)
        if reject_reason:
            reject = _reject_row(entry.index, entry.manifest_id, status_row, reject_reason)
            rejects.append(reject)
            reject_reasons[reject_reason] += 1
            continue
        assert status_row is not None
        _, source_row = source_by_index[entry.index]
        demos.append(_acquisition_row(source_row, status_row))

    missing_status = reject_reasons.get("missing_status", 0)
    if args.require_complete and missing_status:
        raise SystemExit(
            f"{missing_status} source manifest row(s) have no downloader status row; "
            "rerun without --require-complete for a partial snapshot"
        )

    output = {
        "dataset_name": args.dataset_name,
        "schema_version": "counterstrike-1k.hltv_acquisition.v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_manifest": str(args.source_manifest),
        "status_jsonl": [str(path) for path in args.status_jsonl],
        "privacy": {
            "scope": "private acquisition manifest",
            "contains_hltv_match_urls": True,
            "contains_raw_demo_urls": True,
            "contains_steam_ids": False,
            "public_release_note": (
                "Do not publish this manifest. Public split manifests should be "
                "derived from demo_id/clip metadata and must not expose raw demo URLs."
            ),
        },
        "selection_policy": {
            "maps": _unique_values(row.get("map") for row in source_rows),
            "exclude_failed_downloads": True,
            "exclude_multipart_map_demos": True,
            "multipart_pattern": "demo archive member stem ending in -pN or _pN",
        },
        "stats": {
            "source_entries": len(source_entries),
            "status_rows_read": status_count,
            "status_entries_deduplicated": len(status_by_index),
            "selected_full_demo_rows": len(demos),
            "rejected_rows": len(rejects),
            "reject_reasons": dict(sorted(reject_reasons.items())),
            "selected_size_bytes": sum(int(row.get("size_bytes") or 0) for row in demos),
            "selected_by_map": dict(Counter(str(row.get("map")) for row in demos)),
        },
        "demos": demos,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.rejects_jsonl:
        args.rejects_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.rejects_jsonl.open("w", encoding="utf-8") as handle:
            for reject in rejects:
                handle.write(json.dumps(reject, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "rejects_jsonl": str(args.rejects_jsonl) if args.rejects_jsonl else None,
                "selected_full_demo_rows": len(demos),
                "rejected_rows": len(rejects),
                "reject_reasons": dict(sorted(reject_reasons.items())),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join HLTV curation and download ledgers into a full-demo acquisition manifest."
    )
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument(
        "--status-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Downloader status JSONL. May be passed multiple times; later rows win.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rejects-jsonl", type=Path)
    parser.add_argument("--dataset-name", default="counterstrike-1k-dataset")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any source manifest row has no downloader status row.",
    )
    return parser


def _manifest_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        for key in ("matches", "demos", "items", "manifest"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows]
    raise ValueError("source manifest must be a list or an object with matches/demos/items")


def _load_status_rows(paths: list[Path]) -> tuple[dict[int, dict[str, Any]], int]:
    selected: dict[int, tuple[int, int, dict[str, Any]]] = {}
    count = 0
    order = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            count += 1
            order += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_index = _parse_entry_index(row.get("entry_index"))
            if entry_index is None:
                continue
            row["_status_jsonl"] = str(path)
            row["_status_order"] = order
            priority = _status_priority(row)
            current = selected.get(entry_index)
            if current is None or (priority, order) >= (current[0], current[1]):
                selected[entry_index] = (priority, order, row)
    return {index: row for index, (_, _, row) in selected.items()}, count


def _parse_entry_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _status_priority(row: dict[str, Any]) -> int:
    if row.get("status") != "failed":
        return 3
    if row.get("retryable") is True:
        return 0
    if row.get("retryable") is False:
        return 1
    if row.get("demo_id") or row.get("sha256") or row.get("size_bytes"):
        return 1
    if row.get("error_type") in {"ArchiveError", "IntegrityError"}:
        return 1
    return 0


def _reject_reason(status_row: dict[str, Any] | None) -> str | None:
    if status_row is None:
        return "missing_status"
    archive_member = str(status_row.get("archive_member") or "")
    if archive_member and split_demo_part_number(archive_member) is not None:
        return "multipart_archive_member"
    error = str(status_row.get("error") or "").lower()
    if status_row.get("error_type") == "ArchiveError" and "multipart" in error:
        return "multipart_archive"
    if status_row.get("status") == "failed":
        return "failed"
    if not (status_row.get("demo_id") or status_row.get("sha256")):
        return "missing_demo_id"
    if not status_row.get("demo_path"):
        return "missing_demo_path"
    return None


def _reject_row(
    entry_index: int,
    manifest_id: str,
    status_row: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    row = {
        "entry_index": entry_index,
        "manifest_id": manifest_id,
        "reason": reason,
    }
    if status_row is not None:
        row.update(
            {
                "status": status_row.get("status"),
                "archive_member": status_row.get("archive_member"),
                "demo_path": status_row.get("demo_path"),
                "demo_id": status_row.get("demo_id") or status_row.get("sha256"),
                "error_type": status_row.get("error_type"),
                "error": status_row.get("error"),
                "retryable": status_row.get("retryable"),
                "resolved_url": status_row.get("resolved_url"),
                "status_jsonl": status_row.get("_status_jsonl"),
                "status_order": status_row.get("_status_order"),
            }
        )
    return row


def _acquisition_row(source_row: dict[str, Any], status_row: dict[str, Any]) -> dict[str, Any]:
    demo_id = str(status_row.get("demo_id") or status_row.get("sha256"))
    return {
        "entry_index": status_row.get("entry_index"),
        "manifest_id": status_row.get("manifest_id") or source_row.get("manifest_id"),
        "demo_id": demo_id,
        "sha256": status_row.get("sha256") or demo_id,
        "size_bytes": status_row.get("size_bytes"),
        "demo_path": status_row.get("demo_path"),
        "archive_path": status_row.get("archive_path"),
        "archive_member": status_row.get("archive_member"),
        "match_id": status_row.get("match_id") or source_row.get("hltv_match_id"),
        "match_url": status_row.get("match_url") or source_row.get("match_url"),
        "resolved_url": status_row.get("resolved_url"),
        "hltv_demo_id": status_row.get("hltv_demo_id"),
        "map": status_row.get("map_name") or source_row.get("map"),
        "date": source_row.get("date"),
        "event": source_row.get("event"),
        "teams": source_row.get("teams"),
        "score": source_row.get("score"),
        "winner": source_row.get("winner"),
        "best_of": source_row.get("best_of"),
        "hltv_rating_mean": source_row.get("hltv_rating_mean"),
        "team_ranks": source_row.get("team_ranks"),
        "quality": source_row.get("quality"),
        "sources": source_row.get("sources"),
        "download": {
            "status": status_row.get("status", "ok"),
            "skipped": bool(status_row.get("skipped")),
            "parser_checked": bool(status_row.get("parser_checked")),
            "status_jsonl": status_row.get("_status_jsonl"),
            "status_order": status_row.get("_status_order"),
        },
        "curation": source_row,
    }


def _unique_values(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value})


if __name__ == "__main__":
    raise SystemExit(main())
