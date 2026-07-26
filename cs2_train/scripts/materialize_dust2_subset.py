"""Materialize a filtered CounterStrike-1K subset from WebDataset shard offsets.

The output is the direct-file layout consumed by the optimized training loader:
``videos/<resolution>``, ``actions``, ``state``, ``events``, and ``metadata``.
Every selected member is checked against the SHA-256 stored in the public
sample index before it is atomically installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

MEMBER_DESTINATIONS = {
    "mp4": ("videos", "{resolution}", "{sample_key}.mp4"),
    "actions.bin": ("actions", "{sample_key}.actions.bin"),
    "state.bin": ("state", "{sample_key}.state.bin"),
    "events.json": ("events", "{sample_key}.events.json"),
    "json": ("metadata", "{sample_key}.json"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_shard(path_value: str, *, source_root: Path, shard_root: Path) -> Path:
    path = Path(path_value)
    candidates = [
        path if path.is_absolute() else None,
        shard_root / path,
        source_root / path,
        shard_root / path.name,
        source_root / path.name,
    ]
    match = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None and candidate.is_file()
        ),
        None,
    )
    if match is None:
        raise FileNotFoundError(f"could not resolve shard {path_value!r}")
    return match


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def materialize_sample(
    sample_key: str,
    rows: list[dict],
    *,
    source_root: Path,
    shard_root: Path,
    output_root: Path,
    resolution: str,
) -> dict:
    by_suffix = {str(row["member_suffix"]): row for row in rows}
    missing = set(MEMBER_DESTINATIONS) - set(by_suffix)
    if missing:
        raise ValueError(f"{sample_key} is missing indexed members: {sorted(missing)}")

    written_bytes = 0
    skipped = True
    for suffix, destination_parts in MEMBER_DESTINATIONS.items():
        row = by_suffix[suffix]
        destination = output_root.joinpath(
            *[
                part.format(sample_key=sample_key, resolution=resolution)
                for part in destination_parts
            ]
        )
        expected_sha = str(row["member_sha256"])
        if destination.is_file() and sha256_file(destination) == expected_sha:
            written_bytes += destination.stat().st_size
            continue

        skipped = False
        shard = resolve_shard(
            str(row["shard_path"]),
            source_root=source_root,
            shard_root=shard_root,
        )
        offset = int(row["member_offset"])
        length = int(row["member_length"])
        with shard.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise OSError(
                f"{shard}: expected {length} bytes at offset {offset}, got {len(payload)}"
            )
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"{sample_key}.{suffix}: SHA-256 {actual_sha} != {expected_sha}"
            )
        atomic_write(destination, payload)
        written_bytes += len(payload)
    return {"sample_key": sample_key, "bytes": written_bytes, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-name", default="manifest.parquet")
    parser.add_argument("--sample-index-name", default="sample_index_360p.parquet")
    parser.add_argument("--resolution", default="360p")
    parser.add_argument("--map-slug", default="dust2")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest_path = args.source_root / args.manifest_name
    sample_index_path = args.source_root / args.sample_index_name
    manifest = pd.read_parquet(manifest_path)
    selected = manifest[
        (manifest["map_slug"].astype(str) == args.map_slug)
        & manifest["split"].astype(str).isin(args.splits)
    ].copy()
    selected = selected.sort_values(
        ["split", "match_id", "round_idx", "pov_idx"],
        kind="stable",
    ).reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"no rows for map_slug={args.map_slug!r}, splits={args.splits!r}"
        )

    selected_keys = set(selected["sample_key"].astype(str))
    sample_index = pd.read_parquet(sample_index_path)
    sample_index = sample_index[
        sample_index["sample_key"].astype(str).isin(selected_keys)
    ]
    if "resolution" in sample_index.columns:
        sample_index = sample_index[
            sample_index["resolution"].astype(str) == args.resolution
        ]
    grouped = {
        str(sample_key): rows.to_dict("records")
        for sample_key, rows in sample_index.groupby("sample_key", sort=False)
    }
    missing_keys = selected_keys - set(grouped)
    if missing_keys:
        raise ValueError(
            f"{len(missing_keys)} selected samples are absent from {sample_index_path}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    ordered_keys = selected["sample_key"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                lambda key: materialize_sample(
                    key,
                    grouped[key],
                    source_root=args.source_root,
                    shard_root=args.shard_root,
                    output_root=args.output_root,
                    resolution=args.resolution,
                ),
                ordered_keys,
            )
        )

    output_manifest = args.output_root / args.manifest_name
    if len(selected) == len(manifest) and selected_keys == set(
        manifest["sample_key"].astype(str)
    ):
        shutil.copy2(manifest_path, output_manifest)
    else:
        selected.to_parquet(output_manifest, index=False)

    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "code_commit": code_commit,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "sample_index": str(sample_index_path),
        "sample_index_sha256": sha256_file(sample_index_path),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": sha256_file(output_manifest),
        "resolution": args.resolution,
        "map_slug": args.map_slug,
        "splits": args.splits,
        "samples": len(results),
        "bytes": sum(result["bytes"] for result in results),
        "samples_already_verified": sum(result["skipped"] for result in results),
    }
    provenance_path = args.output_root / "materialization_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
