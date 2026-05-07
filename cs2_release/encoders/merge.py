"""Merge row-sharded embedding extraction outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json


def _read_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_embedding_shard(path: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    index_path = path / "embedding_index.parquet"
    npz_path = path / "embeddings.npz"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    index = read_parquet(index_path).reset_index(drop=True)
    payload = np.load(npz_path, allow_pickle=False)
    embeddings = payload["embeddings"].astype(np.float32)
    if len(index) != embeddings.shape[0]:
        raise ValueError(f"{path}: index has {len(index)} rows, embeddings has {embeddings.shape[0]}")

    if "embedding_source_row_id" not in index.columns:
        if "embedding_source_row_id" not in payload:
            raise ValueError(
                f"{path}: missing embedding_source_row_id; rerun extraction with the sharded extractor"
            )
        index = index.copy()
        index["embedding_source_row_id"] = payload["embedding_source_row_id"].astype(np.int64)
    index["embedding_source_row_id"] = index["embedding_source_row_id"].astype(np.int64)
    return index, embeddings, _read_metadata(path)


def merge_embedding_shards(shards: list[Path], out: Path) -> dict:
    indices = []
    embedding_blocks = []
    metadatas = []
    for shard in shards:
        index, embeddings, metadata = load_embedding_shard(shard)
        indices.append(index)
        embedding_blocks.append(embeddings)
        metadatas.append({"path": str(shard), **metadata})

    if not indices:
        raise ValueError("at least one shard is required")

    index = pd.concat(indices, ignore_index=True)
    embeddings = np.concatenate(embedding_blocks, axis=0).astype(np.float32)
    duplicate_mask = index["embedding_source_row_id"].duplicated(keep=False)
    if duplicate_mask.any():
        dupes = sorted(index.loc[duplicate_mask, "embedding_source_row_id"].astype(int).unique().tolist())[:20]
        raise ValueError(f"duplicate embedding_source_row_id values across shards: {dupes}")

    order = np.argsort(index["embedding_source_row_id"].to_numpy(dtype=np.int64), kind="stable")
    index = index.iloc[order].reset_index(drop=True)
    embeddings = embeddings[order]

    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "embedding_index.parquet"
    npz_path = out / "embeddings.npz"
    index.to_parquet(index_path, index=False)
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        sample_key=index["sample_key"].astype(str).to_numpy(),
        eval_window_id=index["eval_window_id"].astype(str).to_numpy(),
        pov_idx=index["pov_idx"].to_numpy(dtype=np.int16),
        embedding_source_row_id=index["embedding_source_row_id"].to_numpy(dtype=np.int64),
    )
    metadata = {
        "rows": int(len(index)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "shards": metadatas,
        "shard_count": int(len(shards)),
        "failed_count": int(sum(int(item.get("failed_count", 0) or 0) for item in metadatas)),
        "index_sha256": dataframe_sha256(index),
        "git_commit": git_commit(),
    }
    encoder_names = sorted({str(item.get("encoder")) for item in metadatas if item.get("encoder")})
    if len(encoder_names) == 1:
        metadata["encoder"] = encoder_names[0]
    write_json(out / "metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True,
                        help="Shard directories containing embedding_index.parquet and embeddings.npz.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = merge_embedding_shards(args.shards, args.out)
    print(args.out)
    print(json.dumps({
        "rows": metadata["rows"],
        "embedding_dim": metadata["embedding_dim"],
        "shard_count": metadata["shard_count"],
        "failed_count": metadata["failed_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
