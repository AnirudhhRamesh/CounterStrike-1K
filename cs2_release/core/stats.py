"""Statistical helpers for release-facing evaluation metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np
import pandas as pd


_MATCH_RE = re.compile(r"(?:^|__)match_?([0-9a-fA-F]+)")


def infer_match_id(value: object) -> str:
    """Best-effort match-id extraction from public eval identifiers."""
    text = str(value)
    match = _MATCH_RE.search(text)
    if match:
        return match.group(1)
    if text.startswith("match_"):
        parts = text.split("__", 1)[0].split("_", 1)
        if len(parts) == 2:
            return parts[1]
    return text


def ensure_query_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``query_match_id`` available for clustered stats."""
    if "query_match_id" in df.columns:
        return df
    out = df.copy()
    if "match_id" in out.columns:
        out["query_match_id"] = out["match_id"].astype(str)
    elif "query_round_id" in out.columns:
        out["query_match_id"] = out["query_round_id"].map(infer_match_id)
    elif "query_eval_window_id" in out.columns:
        out["query_match_id"] = out["query_eval_window_id"].map(infer_match_id)
    elif "eval_window_id" in out.columns:
        out["query_match_id"] = out["eval_window_id"].map(infer_match_id)
    else:
        out["query_match_id"] = "unknown"
    return out


def percentile_ci(values: list[float], *, alpha: float = 0.05) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "low": float(np.quantile(arr, alpha / 2.0)),
        "high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    metric_fn: Callable[[pd.DataFrame], dict[str, float]],
    metrics: list[str],
    n_boot: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    """Bootstrap metrics by resampling clusters with replacement."""
    if df.empty or cluster_col not in df.columns:
        return {}
    clusters = np.asarray(sorted(df[cluster_col].dropna().astype(str).unique()))
    if len(clusters) == 0:
        return {}
    grouped = {cluster: group for cluster, group in df.groupby(cluster_col, sort=False)}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(int(n_boot)):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for draw_idx, cluster in enumerate(chosen):
            part = grouped[str(cluster)].copy()
            part["__bootstrap_draw"] = draw_idx
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        values = metric_fn(boot)
        for metric in metrics:
            value = values.get(metric, float("nan"))
            if np.isfinite(value):
                samples[metric].append(float(value))
    out: dict[str, float] = {"bootstrap_clusters": int(len(clusters)), "bootstrap_samples": int(n_boot)}
    for metric, values in samples.items():
        ci = percentile_ci(values)
        out[f"{metric}_ci_mean"] = ci["mean"]
        out[f"{metric}_ci_low"] = ci["low"]
        out[f"{metric}_ci_high"] = ci["high"]
        out[f"{metric}_ci_std"] = ci["std"]
    return out


def cluster_bootstrap_column_means(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    metrics: list[str],
    n_boot: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    """Bootstrap means of query-level metric columns by cluster."""

    if df.empty or cluster_col not in df.columns:
        return {}
    needed = [cluster_col, *metrics]
    clean = df[needed].dropna(subset=[cluster_col]).copy()
    if clean.empty:
        return {}
    clean[cluster_col] = clean[cluster_col].astype(str)
    clusters = np.asarray(sorted(clean[cluster_col].unique()))
    grouped = {cluster: group for cluster, group in clean.groupby(cluster_col, sort=False)}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(int(n_boot)):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[str(cluster)] for cluster in chosen], ignore_index=True)
        for metric in metrics:
            values = boot[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values):
                samples[metric].append(float(values.mean()))
    out: dict[str, float] = {"bootstrap_clusters": int(len(clusters)), "bootstrap_samples": int(n_boot)}
    for metric, values in samples.items():
        ci = percentile_ci(values)
        out[f"{metric}_ci_mean"] = ci["mean"]
        out[f"{metric}_ci_low"] = ci["low"]
        out[f"{metric}_ci_high"] = ci["high"]
        out[f"{metric}_ci_std"] = ci["std"]
    return out


def cluster_bootstrap_mean_delta(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    baseline_col: str,
    prefix: str,
    n_boot: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    """Bootstrap the mean per-row improvement over a baseline by cluster."""

    if df.empty or cluster_col not in df.columns:
        return {}
    needed = [cluster_col, value_col, baseline_col]
    clean = df[needed].dropna().copy()
    if clean.empty:
        return {}
    clean[cluster_col] = clean[cluster_col].astype(str)
    clusters = np.asarray(sorted(clean[cluster_col].unique()))
    grouped = {cluster: group for cluster, group in clean.groupby(cluster_col, sort=False)}
    observed = float((clean[value_col].to_numpy(dtype=np.float64) - clean[baseline_col].to_numpy(dtype=np.float64)).mean())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [grouped[str(cluster)] for cluster in chosen]
        boot = pd.concat(parts, ignore_index=True)
        delta = boot[value_col].to_numpy(dtype=np.float64) - boot[baseline_col].to_numpy(dtype=np.float64)
        samples.append(float(delta.mean()))
    ci = percentile_ci(samples)
    finite = np.asarray([v for v in samples if np.isfinite(v)], dtype=np.float64)
    p_lte_0 = float((1 + np.sum(finite <= 0.0)) / (len(finite) + 1)) if len(finite) else float("nan")
    return {
        prefix: observed,
        f"{prefix}_ci_mean": ci["mean"],
        f"{prefix}_ci_low": ci["low"],
        f"{prefix}_ci_high": ci["high"],
        f"{prefix}_ci_std": ci["std"],
        f"{prefix}_p_lte_0": p_lte_0,
    }


def one_positive_chance_topk(n_candidates: int, k: int) -> float:
    if n_candidates <= 0:
        return float("nan")
    return float(min(k, n_candidates) / n_candidates)


def hypergeom_hit_probability(num_items: int, num_positive: int, k: int) -> float:
    if num_items <= 0 or num_positive <= 0:
        return 0.0
    k = min(k, num_items)
    num_negative = num_items - num_positive
    if k > num_negative:
        return 1.0
    return float(1.0 - math.comb(num_negative, k) / math.comb(num_items, k))
