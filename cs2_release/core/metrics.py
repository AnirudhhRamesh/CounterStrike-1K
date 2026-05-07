"""Small metric helpers without extra dependencies."""

from __future__ import annotations

import math

import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_norm @ b_norm.T


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    labels = np.asarray(labels).astype(np.int32)
    pred = (np.asarray(scores) >= threshold).astype(np.int32)
    return float((pred == labels).mean()) if len(labels) else float("nan")


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    ranked = labels[order].astype(bool)
    hit_count = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    precision_at_hits = hit_count[ranked] / ranks[ranked]
    return float(precision_at_hits.mean()) if len(precision_at_hits) else float("nan")


def retrieval_metrics(groups: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Compute retrieval metrics from ``[(scores, labels), ...]`` groups."""

    if not groups:
        return {
            "queries": 0,
            "top1": float("nan"),
            "top5": float("nan"),
            "mrr": float("nan"),
            "mean_positive_rank": float("nan"),
            "mean_candidates": float("nan"),
        }
    top1 = []
    top5 = []
    rr = []
    positive_ranks = []
    candidates = []
    for scores, labels in groups:
        scores = np.asarray(scores)
        labels = np.asarray(labels).astype(bool)
        order = np.argsort(-scores)
        ranked_labels = labels[order]
        pos = np.flatnonzero(ranked_labels)
        if len(pos) == 0:
            continue
        rank = int(pos[0]) + 1
        top1.append(rank == 1)
        top5.append(rank <= 5)
        rr.append(1.0 / rank)
        positive_ranks.append(rank)
        candidates.append(len(scores))
    return {
        "queries": int(len(top1)),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "top5": float(np.mean(top5)) if top5 else float("nan"),
        "mrr": float(np.mean(rr)) if rr else float("nan"),
        "mean_positive_rank": float(np.mean(positive_ranks)) if positive_ranks else float("nan"),
        "mean_candidates": float(np.mean(candidates)) if candidates else float("nan"),
    }


def _chance_hit_at_k(n_candidates: int, n_positive: int, k: int) -> float:
    if n_candidates <= 0 or n_positive <= 0:
        return float("nan")
    k = min(k, n_candidates)
    n_negative = n_candidates - n_positive
    if k > n_negative:
        return 1.0
    return float(1.0 - (math.comb(n_negative, k) / math.comb(n_candidates, k)))


def multipositive_retrieval_metrics(
    groups: list[tuple[np.ndarray, np.ndarray]],
    *,
    recall_ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval metrics where each candidate set may contain many positives."""

    if not groups:
        out = {
            "queries": 0,
            "mrr": float("nan"),
            "map": float("nan"),
            "mean_candidates": float("nan"),
            "mean_positives": float("nan"),
            "mean_positive_fraction": float("nan"),
        }
        for k in recall_ks:
            out[f"hit@{k}"] = float("nan")
            out[f"recall@{k}"] = float("nan")
            out[f"chance_hit@{k}"] = float("nan")
            out[f"chance_recall@{k}"] = float("nan")
        return out

    rr = []
    ap = []
    candidates = []
    positives = []
    hits = {k: [] for k in recall_ks}
    recalls = {k: [] for k in recall_ks}
    chance_hits = {k: [] for k in recall_ks}
    chance_recalls = {k: [] for k in recall_ks}
    for scores, labels in groups:
        scores = np.asarray(scores)
        labels = np.asarray(labels).astype(bool)
        if len(scores) == 0 or not labels.any():
            continue
        order = np.argsort(-scores)
        ranked = labels[order]
        n = int(len(ranked))
        p = int(ranked.sum())
        pos = np.flatnonzero(ranked)
        first_rank = int(pos[0]) + 1
        rr.append(1.0 / first_rank)
        hit_count = np.cumsum(ranked)
        ranks = np.arange(1, n + 1)
        ap.append(float((hit_count[ranked] / ranks[ranked]).mean()))
        candidates.append(n)
        positives.append(p)
        for k in recall_ks:
            kk = min(int(k), n)
            found = int(ranked[:kk].sum())
            hits[k].append(found > 0)
            recalls[k].append(found / p)
            chance_hits[k].append(_chance_hit_at_k(n, p, kk))
            chance_recalls[k].append(kk / n)

    out = {
        "queries": int(len(rr)),
        "mrr": float(np.mean(rr)) if rr else float("nan"),
        "map": float(np.mean(ap)) if ap else float("nan"),
        "mean_candidates": float(np.mean(candidates)) if candidates else float("nan"),
        "mean_positives": float(np.mean(positives)) if positives else float("nan"),
        "mean_positive_fraction": float(np.mean(np.array(positives) / np.array(candidates)))
        if candidates else float("nan"),
    }
    for k in recall_ks:
        out[f"hit@{k}"] = float(np.mean(hits[k])) if hits[k] else float("nan")
        out[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else float("nan")
        out[f"chance_hit@{k}"] = float(np.mean(chance_hits[k])) if chance_hits[k] else float("nan")
        out[f"chance_recall@{k}"] = float(np.mean(chance_recalls[k])) if chance_recalls[k] else float("nan")
    return out


def finite_row_mask(embeddings: np.ndarray) -> np.ndarray:
    return np.isfinite(embeddings).all(axis=1)
