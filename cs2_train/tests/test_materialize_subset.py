from __future__ import annotations

import hashlib
from pathlib import Path

from cs2_train.scripts.materialize_dust2_subset import (
    MEMBER_DESTINATIONS,
    materialize_sample,
)


def test_materialize_sample_seeks_and_verifies_indexed_members(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    shard_root = tmp_path / "shards"
    output_root = tmp_path / "output"
    source_root.mkdir()
    shard_root.mkdir()
    shard_path = shard_root / "release-00000.tar"

    rows = []
    offset = 0
    shard_payload = bytearray()
    for suffix in MEMBER_DESTINATIONS:
        payload = f"payload-{suffix}".encode()
        shard_payload.extend(payload)
        rows.append(
            {
                "member_suffix": suffix,
                "shard_path": shard_path.name,
                "member_offset": offset,
                "member_length": len(payload),
                "member_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        offset += len(payload)
    shard_path.write_bytes(shard_payload)

    result = materialize_sample(
        "sample",
        rows,
        source_root=source_root,
        shard_root=shard_root,
        output_root=output_root,
        resolution="360p",
    )
    assert result["skipped"] is False
    assert (
        output_root / "videos" / "360p" / "sample.mp4"
    ).read_bytes() == b"payload-mp4"
    assert (
        output_root / "actions" / "sample.actions.bin"
    ).read_bytes() == b"payload-actions.bin"

    repeated = materialize_sample(
        "sample",
        rows,
        source_root=source_root,
        shard_root=shard_root,
        output_root=output_root,
        resolution="360p",
    )
    assert repeated["skipped"] is True
