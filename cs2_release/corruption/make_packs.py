"""Build 10-POV real-vs-corrupted video pack manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json


def _complete_groups(windows: pd.DataFrame, split: str) -> dict[str, pd.DataFrame]:
    df = windows[windows["split"] == split].copy()
    groups = {}
    for eval_window_id, group in df.groupby("eval_window_id", sort=False):
        if len(group) != 10:
            continue
        povs = sorted(group["pov_idx"].astype(int).tolist())
        if povs != list(range(10)):
            continue
        groups[str(eval_window_id)] = group.sort_values("pov_idx").copy()
    return groups


def build_corruption_sets(
    windows: pd.DataFrame,
    *,
    split: str,
    severities: list[int],
    negatives_per_positive: int,
    max_positives: int | None,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = _complete_groups(windows, split)
    group_items = list(groups.items())
    if max_positives is not None and len(group_items) > max_positives:
        selected = np.sort(rng.choice(np.arange(len(group_items)), size=max_positives, replace=False))
        group_items = [group_items[int(i)] for i in selected]

    by_context: dict[tuple[str, str], list[tuple[str, pd.DataFrame]]] = {}
    for eval_window_id, group in groups.items():
        first = group.iloc[0]
        by_context.setdefault((str(first["map_slug"]), str(first["phase_bucket"])), []).append((eval_window_id, group))

    rows: list[dict] = []
    for eval_window_id, group in group_items:
        first = group.iloc[0]
        member_eval_ids = [eval_window_id] * 10
        member_povs = group["pov_idx"].astype(int).tolist()
        rows.append({
            "pack_id": f"{eval_window_id}__real",
            "label": 1,
            "severity": 0,
            "split": split,
            "map_slug": str(first["map_slug"]),
            "phase_bucket": str(first["phase_bucket"]),
            "base_eval_window_id": eval_window_id,
            "member_eval_window_ids": json.dumps(member_eval_ids),
            "member_pov_idx": json.dumps(member_povs),
            "negative_policy": "none",
        })

        context_pool = [
            item for item in by_context.get((str(first["map_slug"]), str(first["phase_bucket"])), [])
            if item[0] != eval_window_id
        ]
        if not context_pool:
            context_pool = [
                item for item in groups.items()
                if item[0] != eval_window_id and str(item[1].iloc[0]["map_slug"]) == str(first["map_slug"])
            ]
        if not context_pool:
            continue
        for severity in severities:
            k = int(severity)
            if k <= 0 or k > 9:
                continue
            for neg_idx in range(negatives_per_positive):
                replacement_eval_id, _ = context_pool[int(rng.integers(0, len(context_pool)))]
                replace_povs = sorted(rng.choice(np.arange(10), size=k, replace=False).astype(int).tolist())
                corrupt_eval_ids = list(member_eval_ids)
                for pov in replace_povs:
                    corrupt_eval_ids[pov] = replacement_eval_id
                rows.append({
                    "pack_id": f"{eval_window_id}__corrupt_s{k}_{neg_idx:02d}",
                    "label": 0,
                    "severity": k,
                    "split": split,
                    "map_slug": str(first["map_slug"]),
                    "phase_bucket": str(first["phase_bucket"]),
                    "base_eval_window_id": eval_window_id,
                    "member_eval_window_ids": json.dumps(corrupt_eval_ids),
                    "member_pov_idx": json.dumps(member_povs),
                    "negative_policy": "same_map_phase_replace_same_pov_slots",
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-positives", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = read_parquet(args.windows)
    packs = build_corruption_sets(
        windows,
        split=args.split,
        severities=args.severities,
        negatives_per_positive=args.negatives_per_positive,
        max_positives=args.max_positives,
        seed=args.seed,
    )
    if packs.empty:
        raise RuntimeError("no corruption packs were produced")
    out_path = args.out / f"corruption_{args.split}.parquet" if args.out.suffix == "" else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packs.to_parquet(out_path, index=False)
    write_json(out_path.with_suffix(".metadata.json"), {
        "rows": int(len(packs)),
        "positives": int((packs["label"] == 1).sum()),
        "negatives": int((packs["label"] == 0).sum()),
        "split": args.split,
        "severities": args.severities,
        "negatives_per_positive": args.negatives_per_positive,
        "max_positives": args.max_positives,
        "seed": args.seed,
        "packs_sha256": dataframe_sha256(packs),
        "git_commit": git_commit(),
    })
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
