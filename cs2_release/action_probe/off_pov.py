"""Quantify action events missed by a randomly selected single POV.

For each synchronized 10-POV window, the full multi-POV label is positive if
any POV contains the action. The single-POV miss rate is the expected
probability that a uniformly sampled POV from that same window would not show
the action, conditioned on the multi-POV window being positive.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.action_probe.coverage import DEFAULT_ACTIONS
from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.stats import percentile_ci


def _window_table(labels: pd.DataFrame, action: str) -> pd.DataFrame:
    col = f"label_{action}"
    grouped = labels.groupby("eval_window_id", sort=False)
    table = pd.DataFrame({
        "split": grouped["split"].first().astype(str),
        "match_id": (
            grouped["match_id"].first().astype(str)
            if "match_id" in labels.columns
            else grouped["eval_window_id"].first().astype(str)
        ),
        "pov_count": grouped["pov_idx"].nunique().astype(int),
        "single_pov_detection": grouped[col].mean().astype(float),
        "multi_pov_event": (grouped[col].sum().astype(int) > 0).astype(float),
    }).reset_index(drop=True)
    table["single_pov_miss_given_event"] = np.where(
        table["multi_pov_event"].to_numpy(dtype=bool),
        1.0 - table["single_pov_detection"].to_numpy(dtype=np.float64),
        np.nan,
    )
    return table


def _summarize(table: pd.DataFrame) -> dict[str, float | int]:
    positives = table[table["multi_pov_event"] > 0]
    single_prev = float(table["single_pov_detection"].mean()) if len(table) else float("nan")
    multi_prev = float(table["multi_pov_event"].mean()) if len(table) else float("nan")
    detect_given_event = (
        float(positives["single_pov_detection"].mean()) if len(positives) else float("nan")
    )
    miss_given_event = 1.0 - detect_given_event if np.isfinite(detect_given_event) else float("nan")
    return {
        "windows": int(len(table)),
        "positive_windows": int(len(positives)),
        "single_pov_prevalence": single_prev,
        "multi_pov_event_prevalence": multi_prev,
        "multi_vs_single_prevalence_lift": (
            multi_prev / single_prev if np.isfinite(single_prev) and single_prev > 0 else float("inf")
        ),
        "single_pov_detection_rate_given_event": detect_given_event,
        "single_pov_miss_rate_given_event": miss_given_event,
        "mean_povs_per_window": float(table["pov_count"].mean()) if len(table) else float("nan"),
    }


def _bootstrap(table: pd.DataFrame, *, samples: int, seed: int) -> dict[str, float | int]:
    if samples <= 0 or "match_id" not in table.columns or table.empty:
        return {}
    match_ids = sorted(table["match_id"].astype(str).unique())
    by_match = {match_id: group for match_id, group in table.groupby("match_id", sort=False)}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "single_pov_prevalence": [],
        "multi_pov_event_prevalence": [],
        "single_pov_miss_rate_given_event": [],
    }
    for _ in range(int(samples)):
        chosen = rng.choice(match_ids, size=len(match_ids), replace=True)
        boot = pd.concat([by_match[str(match_id)] for match_id in chosen], ignore_index=True)
        summary = _summarize(boot)
        for key in values:
            value = float(summary[key])
            if np.isfinite(value):
                values[key].append(value)
    out: dict[str, float | int] = {
        "bootstrap_matches": int(len(match_ids)),
        "bootstrap_samples": int(samples),
    }
    for key, samples_for_key in values.items():
        ci = percentile_ci(samples_for_key)
        out[f"{key}_ci_low"] = ci["low"]
        out[f"{key}_ci_high"] = ci["high"]
        out[f"{key}_ci_std"] = ci["std"]
    return out


def compute_visibility(
    labels: pd.DataFrame,
    *,
    actions: list[str],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    required = {"split", "eval_window_id", "pov_idx"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels file is missing required columns: {missing}")
    rows = []
    for action in actions:
        col = f"label_{action}"
        if col not in labels.columns:
            continue
        window_table = _window_table(labels, action)
        for split, split_table in window_table.groupby("split", sort=True):
            record = {"split": str(split), "label": action, **_summarize(split_table)}
            salt = zlib.crc32(f"{split}:{action}".encode("utf-8"))
            record.update(_bootstrap(
                split_table,
                samples=bootstrap_samples,
                seed=int((seed + salt) % (2**32 - 1)),
            ))
            rows.append(record)
    if not rows:
        raise RuntimeError("no off-POV action visibility rows were produced")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", default=DEFAULT_ACTIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    labels = read_parquet(args.labels)
    visibility = compute_visibility(
        labels,
        actions=list(args.actions),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    if args.out.suffix:
        out_path = args.out
        out_dir = out_path.parent
    else:
        out_dir = args.out
        out_path = out_dir / "off_pov_action_visibility.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    visibility.to_parquet(out_path, index=False)
    visibility.to_csv(out_path.with_suffix(".csv"), index=False)
    write_json(out_path.with_suffix(".metrics.json"), {
        "labels_path": str(args.labels),
        "labels_sha256": dataframe_sha256(labels),
        "visibility_sha256": dataframe_sha256(visibility),
        "actions": list(args.actions),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "rows": int(len(visibility)),
        "git_commit": git_commit(),
    })
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
