"""Embedding table loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import read_parquet


def load_embedding_table(embedding_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    embedding_dir = Path(embedding_dir)
    index_path = embedding_dir / "embedding_index.parquet"
    npz_path = embedding_dir / "embeddings.npz"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    index = read_parquet(index_path).reset_index(drop=True)
    payload = np.load(npz_path, allow_pickle=False)
    embeddings = payload["embeddings"].astype(np.float32)
    if len(index) != embeddings.shape[0]:
        raise ValueError(
            f"embedding row mismatch: index has {len(index)}, embeddings has {embeddings.shape[0]}"
        )
    index = index.copy()
    index["embedding_row_id"] = np.arange(len(index), dtype=np.int64)
    return index, embeddings
