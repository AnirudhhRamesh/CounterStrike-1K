"""Shared helpers for 10-POV corruption detection."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PackArrays:
    features: np.ndarray
    labels: np.ndarray
    severities: np.ndarray
    pack_ids: list[str]


def parse_json_list(value) -> list:
    if isinstance(value, list):
        return value
    return json.loads(str(value))


def pack_feature_from_embeddings(member_embeddings: np.ndarray) -> np.ndarray:
    """Convert a 10xD embedding pack into a compact consistency feature."""

    emb = member_embeddings.astype(np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    mean = emb.mean(axis=0)
    std = emb.std(axis=0)
    sim = emb @ emb.T
    tri = sim[np.triu_indices(sim.shape[0], k=1)]
    summary = np.array([
        float(tri.mean()),
        float(tri.std()),
        float(tri.min()),
        float(tri.max()),
    ], dtype=np.float32)
    return np.concatenate([mean, std, summary]).astype(np.float32)


def attach_pack_embedding_rows(packs: pd.DataFrame, embedding_index: pd.DataFrame) -> pd.DataFrame:
    row_map = {
        (str(row["eval_window_id"]), int(row["pov_idx"])): int(row["embedding_row_id"])
        for _, row in embedding_index.iterrows()
    }
    rows = []
    for _, pack in packs.iterrows():
        eval_window_ids = parse_json_list(pack["member_eval_window_ids"])
        povs = [int(v) for v in parse_json_list(pack["member_pov_idx"])]
        member_rows = []
        missing = False
        for eval_window_id, pov_idx in zip(eval_window_ids, povs, strict=True):
            key = (str(eval_window_id), int(pov_idx))
            if key not in row_map:
                missing = True
                break
            member_rows.append(row_map[key])
        if missing:
            continue
        row = pack.to_dict()
        row["member_embedding_row_ids"] = json.dumps(member_rows)
        rows.append(row)
    return pd.DataFrame(rows)


def build_pack_arrays(packs: pd.DataFrame, embeddings: np.ndarray) -> PackArrays:
    features = []
    labels = []
    severities = []
    pack_ids = []
    finite = np.isfinite(embeddings).all(axis=1)
    for _, pack in packs.iterrows():
        row_ids = np.array(parse_json_list(pack["member_embedding_row_ids"]), dtype=np.int64)
        if len(row_ids) != 10 or not finite[row_ids].all():
            continue
        features.append(pack_feature_from_embeddings(embeddings[row_ids]))
        labels.append(int(pack["label"]))
        severities.append(int(pack["severity"]))
        pack_ids.append(str(pack["pack_id"]))
    if not features:
        return PackArrays(
            features=np.empty((0, 0), dtype=np.float32),
            labels=np.empty((0,), dtype=np.float32),
            severities=np.empty((0,), dtype=np.int32),
            pack_ids=[],
        )
    return PackArrays(
        features=np.stack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.float32),
        severities=np.array(severities, dtype=np.int32),
        pack_ids=pack_ids,
    )
