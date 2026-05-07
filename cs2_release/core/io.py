"""Shared public-artifact I/O helpers for CounterStrike-1K evaluations."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


VIDEO_SUFFIX = "mp4"


@dataclass(frozen=True)
class DatasetRoots:
    """Local release roots used by the evaluation suite."""

    root: Path
    shard_root: Path
    resolution: str

    @classmethod
    def from_args(
        cls,
        *,
        root: str | Path,
        shard_root: str | Path | None = None,
        resolution: str = "360p",
    ) -> "DatasetRoots":
        if resolution not in {"360p", "720p"}:
            raise ValueError("resolution must be '360p' or '720p'")
        root_path = Path(root)
        return cls(
            root=root_path,
            shard_root=Path(shard_root) if shard_root is not None else root_path,
            resolution=resolution,
        )


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def read_release_parquet(root: str | Path, relative_path: str | Path) -> pd.DataFrame:
    """Read a release parquet from local disk or opt-in maintainer S3 storage."""

    root = Path(root)
    relative = Path(relative_path)
    local = root / relative
    if local.exists():
        return pd.read_parquet(local)
    bucket = os.environ.get("CS2_EVAL_S3_BUCKET")
    if not bucket:
        raise FileNotFoundError(local)
    key = str(relative).lstrip("/")
    if os.environ.get("CS2_EVAL_S3_PREFIX"):
        key = os.environ["CS2_EVAL_S3_PREFIX"].rstrip("/") + "/" + key
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for CS2_EVAL_S3_BUCKET parquet reads") from exc
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Return a stable hash for a small/medium DataFrame."""

    view = df[columns] if columns is not None else df
    payload = view.sort_index(axis=1).to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - best-effort provenance only.
        return None


def load_release_tables(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    manifest = read_parquet(root / "manifest.parquet")
    round_index = read_parquet(root / "round_index.parquet")
    return manifest, round_index


def load_subset_keys(root: str | Path, subset: str | None) -> set[str] | None:
    if not subset:
        return None
    subset_path = Path(root) / "subsets" / f"{subset}.parquet"
    if not subset_path.exists():
        raise FileNotFoundError(subset_path)
    subset_df = read_parquet(subset_path)
    if "sample_key" not in subset_df.columns:
        raise ValueError(f"{subset_path} does not contain a sample_key column")
    return set(subset_df["sample_key"].astype(str).tolist())


def filter_manifest_for_subset(
    manifest: pd.DataFrame,
    *,
    root: str | Path,
    subset: str | None,
) -> pd.DataFrame:
    keys = load_subset_keys(root, subset)
    if keys is None:
        return manifest
    return manifest[manifest["sample_key"].astype(str).isin(keys)].reset_index(drop=True)


def resolve_shard_path(row: pd.Series, roots: DatasetRoots) -> Path:
    shard_path = str(row["shard_path"])
    path = Path(shard_path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    candidates.extend([
        roots.shard_root / shard_path,
        roots.root / shard_path,
        roots.shard_root / path.name,
        roots.root / path.name,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not resolve shard path "
        + shard_path
        + " from candidates "
        + ", ".join(str(c) for c in candidates)
    )


def _s3_range_read(
    *,
    row: pd.Series,
    roots: DatasetRoots,
    offset: int,
    length: int,
) -> bytes | None:
    """Best-effort S3 byte-range read for private smoke tests.

    Public/reviewer workflows should use local HF snapshots. This hook exists so
    maintainers can validate freshly packed shards in S3 without downloading
    multi-GB tar files. It is opt-in via ``CS2_EVAL_S3_BUCKET``.
    """

    bucket = os.environ.get("CS2_EVAL_S3_BUCKET")
    if not bucket:
        return None
    key = str(row["shard_path"])
    if key.startswith("s3://"):
        without_scheme = key.removeprefix("s3://")
        bucket, key = without_scheme.split("/", 1)
    elif os.environ.get("CS2_EVAL_S3_PREFIX"):
        key = os.environ["CS2_EVAL_S3_PREFIX"].rstrip("/") + "/" + key.lstrip("/")
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for CS2_EVAL_S3_BUCKET range reads") from exc
    s3 = boto3.client("s3")
    end = offset + length - 1
    obj = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{end}")
    payload = obj["Body"].read()
    if len(payload) != length:
        raise IOError(f"s3://{bucket}/{key}: expected {length} range bytes, got {len(payload)}")
    return payload


def direct_video_path(sample_key: str, roots: DatasetRoots) -> Path | None:
    candidates = [
        roots.root / "videos" / roots.resolution / f"{sample_key}.mp4",
        roots.root / "videos" / f"{sample_key}.mp4",
        roots.root / roots.resolution / f"{sample_key}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_video_bytes(
    sample_key: str,
    *,
    roots: DatasetRoots,
    sample_index: pd.DataFrame | None,
    verify_sha256: bool = False,
) -> bytes:
    """Read one MP4 from direct sample files or uncompressed WDS byte offsets."""

    direct = direct_video_path(sample_key, roots)
    if direct is not None:
        return direct.read_bytes()

    if sample_index is None:
        index_path = roots.root / "sample_index.parquet"
        if not index_path.exists():
            raise FileNotFoundError(
                f"no direct MP4 for {sample_key} and no sample_index.parquet"
            )
        sample_index = read_parquet(index_path)

    rows = sample_index[
        (sample_index["sample_key"].astype(str) == str(sample_key))
        & (sample_index["resolution"].astype(str) == roots.resolution)
        & (sample_index["member_suffix"].astype(str) == VIDEO_SUFFIX)
    ]
    if rows.empty:
        raise ValueError(f"sample_index has no {roots.resolution} MP4 for {sample_key}")
    row = rows.iloc[0]
    offset = int(row["member_offset"])
    length = int(row["member_length"])
    try:
        shard = resolve_shard_path(row, roots)
    except FileNotFoundError:
        payload = _s3_range_read(row=row, roots=roots, offset=offset, length=length)
        if payload is None:
            raise
    else:
        with shard.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
    if len(payload) != length:
        raise IOError(f"{shard}: expected {length} bytes at offset {offset}, got {len(payload)}")
    expected_sha = str(row.get("member_sha256") or "")
    if verify_sha256 and expected_sha:
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"sha256 mismatch for {sample_key}.mp4: {actual_sha} != {expected_sha}")
    return payload


def read_member_bytes(
    sample_key: str,
    member_suffix: str,
    *,
    roots: DatasetRoots,
    sample_index: pd.DataFrame | None,
    verify_sha256: bool = False,
) -> bytes:
    """Read one non-video member from direct files or WDS byte offsets."""

    candidates = [
        roots.root / member_suffix / f"{sample_key}.{member_suffix}",
        roots.root / "actions" / "v12" / f"{sample_key}.{member_suffix}",
        roots.root / "actions" / f"{sample_key}.{member_suffix}",
        roots.root / "state" / "v12" / f"{sample_key}.{member_suffix}",
        roots.root / "state" / f"{sample_key}.{member_suffix}",
        roots.root / "events" / "v12" / f"{sample_key}.{member_suffix}",
        roots.root / "events" / f"{sample_key}.{member_suffix}",
        roots.root / "metadata" / "v12" / f"{sample_key}.{member_suffix}",
        roots.root / "metadata" / f"{sample_key}.{member_suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_bytes()

    if sample_index is None:
        index_path = roots.root / f"sample_index_{roots.resolution}.parquet"
        if not index_path.exists():
            index_path = roots.root / "sample_index.parquet"
        if not index_path.exists():
            raise FileNotFoundError(f"no sample index for {sample_key}.{member_suffix}")
        sample_index = read_parquet(index_path)

    rows = sample_index[
        (sample_index["sample_key"].astype(str) == str(sample_key))
        & (sample_index["resolution"].astype(str) == roots.resolution)
        & (sample_index["member_suffix"].astype(str) == member_suffix)
    ]
    if rows.empty:
        raise ValueError(f"sample_index has no {roots.resolution} {member_suffix} for {sample_key}")
    row = rows.iloc[0]
    offset = int(row["member_offset"])
    length = int(row["member_length"])
    try:
        shard = resolve_shard_path(row, roots)
    except FileNotFoundError:
        payload = _s3_range_read(row=row, roots=roots, offset=offset, length=length)
        if payload is None:
            raise
    else:
        with shard.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
    if len(payload) != length:
        raise IOError(f"expected {length} bytes at offset {offset}, got {len(payload)}")
    expected_sha = str(row.get("member_sha256") or "")
    if verify_sha256 and expected_sha:
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"sha256 mismatch for {sample_key}.{member_suffix}: {actual_sha} != {expected_sha}"
            )
    return payload
