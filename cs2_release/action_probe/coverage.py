"""Measure how multi-POV capture increases action-event coverage.

This is a dataset-level evaluation: given per-POV action labels for a
synchronized eval window, compute the expected probability that at least one
action instance is visible when sampling k POVs from the 10 synchronized POVs.

The result is a direct single-POV vs multi-POV comparison and does not train a
model. It is useful for paper tables because it quantifies a core advantage of
CounterStrike-1K over single-view gameplay corpora.
"""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json
from cs2_release.core.stats import percentile_ci


DEFAULT_ACTIONS = [
    "FIRE",
    "RIGHTCLICK",
    "RELOAD",
    "JUMP",
    "DUCK",
    "WALK",
    "MOUSE_MOVE",
    "INSPECT",
]


def _coverage_probability(num_povs: int, num_positive_povs: int, k: int) -> float:
    """P(at least one positive POV in k samples without replacement)."""
    if k <= 0:
        return 0.0
    if num_positive_povs <= 0:
        return 0.0
    if k >= num_povs:
        return 1.0
    num_negative = num_povs - num_positive_povs
    if k > num_negative:
        return 1.0
    return 1.0 - (math.comb(num_negative, k) / math.comb(num_povs, k))


def compute_coverage(
    labels: pd.DataFrame,
    *,
    actions: list[str],
    k_values: list[int],
    bootstrap_samples: int = 0,
    seed: int = 123,
) -> pd.DataFrame:
    required = {"split", "eval_window_id", "pov_idx"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels file is missing required columns: {missing}")

    rows: list[dict] = []
    for split, split_df in labels.groupby("split", sort=True):
        grouped = split_df.groupby("eval_window_id", sort=False)
        num_windows = int(grouped.ngroups)
        if num_windows == 0:
            continue
        pov_counts = grouped["pov_idx"].nunique()
        if "match_id" in split_df.columns:
            window_match_ids = grouped["match_id"].first().astype(str)
        else:
            window_match_ids = grouped["eval_window_id"].first().astype(str)
        mean_povs = float(pov_counts.mean())

        for action in actions:
            col = f"label_{action}"
            if col not in split_df.columns:
                continue
            pos_counts = grouped[col].sum().astype(int)
            per_pov_prevalence = float(split_df[col].mean())
            any_10pov_prevalence = float((pos_counts > 0).mean())
            window_stats = pd.DataFrame({
                "match_id": window_match_ids,
                "pov_count": pov_counts.astype(int),
                "pos_count": pos_counts.astype(int),
            }).reset_index(drop=True)
            by_match = {
                str(match_id): group[["pov_count", "pos_count"]].to_numpy(dtype=np.int16)
                for match_id, group in window_stats.groupby("match_id", sort=False)
            }
            match_ids = sorted(by_match)
            for k in k_values:
                probs = [
                    _coverage_probability(int(n), int(m), int(k))
                    for n, m in zip(pov_counts.to_numpy(), pos_counts.to_numpy(), strict=True)
                ]
                coverage = float(sum(probs) / len(probs))
                record = {
                    "split": str(split),
                    "label": action,
                    "k_povs": int(k),
                    "coverage": coverage,
                    "per_pov_prevalence": per_pov_prevalence,
                    "any_10pov_prevalence": any_10pov_prevalence,
                    "lift_vs_single_pov": (
                        coverage / per_pov_prevalence if per_pov_prevalence > 0 else float("inf")
                    ),
                    "num_windows": num_windows,
                    "mean_povs_per_window": mean_povs,
                }
                if bootstrap_samples > 0 and "match_id" in split_df.columns:
                    boot_values = []
                    salt = zlib.crc32(f"{split}:{action}:{k}".encode("utf-8"))
                    boot_rng = np.random.default_rng(int((seed + salt) % (2**32 - 1)))
                    for _ in range(int(bootstrap_samples)):
                        chosen = boot_rng.choice(match_ids, size=len(match_ids), replace=True)
                        boot_arr = np.concatenate([by_match[str(match_id)] for match_id in chosen], axis=0)
                        probs = [
                            _coverage_probability(int(n), int(m), int(k))
                            for n, m in zip(boot_arr[:, 0], boot_arr[:, 1], strict=True)
                        ]
                        boot_values.append(float(sum(probs) / len(probs)) if probs else float("nan"))
                    ci = percentile_ci(boot_values)
                    record["coverage_ci_low"] = ci["low"]
                    record["coverage_ci_high"] = ci["high"]
                    record["coverage_ci_std"] = ci["std"]
                    record["bootstrap_matches"] = len(match_ids)
                    record["bootstrap_samples"] = int(bootstrap_samples)
                rows.append(record)
    if not rows:
        raise RuntimeError("no action coverage rows were produced")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="action_probe_labels.parquet from make_action_probe_labels")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory or .parquet/.csv path")
    parser.add_argument("--actions", nargs="+", default=DEFAULT_ACTIONS)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    labels = read_parquet(args.labels)
    coverage = compute_coverage(
        labels,
        actions=list(args.actions),
        k_values=list(args.k_values),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    if args.out.suffix:
        out_path = args.out
        out_dir = out_path.parent
    else:
        out_dir = args.out
        out_path = out_dir / "multipov_action_coverage.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(out_path, index=False)
    coverage.to_csv(out_path.with_suffix(".csv"), index=False)
    write_json(out_path.with_suffix(".metrics.json"), {
        "labels_path": str(args.labels),
        "labels_sha256": dataframe_sha256(labels),
        "coverage_sha256": dataframe_sha256(coverage),
        "actions": list(args.actions),
        "k_values": [int(k) for k in args.k_values],
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "rows": int(len(coverage)),
        "git_commit": git_commit(),
    })
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
