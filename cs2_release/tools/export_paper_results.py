"""Print compact paper table rows from CounterStrike-1K eval outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def video_row(run_dir: Path, *, label: str) -> str:
    windows = read_json(run_dir / "windows" / "eval_windows.metadata.json")
    retrieval = read_json(run_dir / "retrieval" / "metrics_retrieval.json")
    corruption = read_json(run_dir / "corruption" / "test" / "metrics_corruption.json")
    rounds_per_split = int(windows["rounds"]) // max(1, len(windows.get("splits", [])))
    return (
        f"{label} & {rounds_per_split:,} & {int(windows['rows']):,} & "
        f"{fmt(retrieval.get('top1'))} & {fmt(retrieval.get('top5'))} & "
        f"{fmt(retrieval.get('mrr'))} & {fmt(corruption.get('auc'))} \\\\"
    )


def video_extended_row(run_dir: Path, *, label: str) -> str:
    windows = read_json(run_dir / "windows" / "eval_windows.metadata.json")
    retrieval = read_json(run_dir / "retrieval" / "metrics_retrieval.json")
    multipositive = read_json(run_dir / "multipositive_retrieval" / "metrics_retrieval.json")
    spatial = read_json(run_dir / "spatial_retrieval" / "metrics_retrieval.json")
    action = read_json(run_dir / "action_probe" / "metrics_action_probe.json")
    rounds_per_split = int(windows["rounds"]) // max(1, len(windows.get("splits", [])))
    return (
        f"{label} & {rounds_per_split:,} & {int(windows['rows']):,} & "
        f"{fmt(retrieval.get('top1'))} & {fmt(multipositive.get('hit@1'))} & "
        f"{fmt(multipositive.get('chance_hit@1'))} & {fmt(spatial.get('hit@1'))} & "
        f"{fmt(spatial.get('chance_hit@1'))} & {fmt(action.get('test', {}).get('macro_auc'))} \\\\"
    )


def diamond_row(run_dir: Path, *, label: str) -> str:
    train_log = read_json(run_dir / "config.json")
    ckpt_metrics = run_dir / "metrics_train.json"
    if ckpt_metrics.exists():
        metrics = read_json(ckpt_metrics)
        best_mse = metrics.get("best_val_mse")
        psnr = metrics.get("best_val_psnr_db")
    else:
        best_mse, psnr = best_validation_from_log(run_dir / "train.log")
    return (
        f"{label} & {int(train_log.get('max_train_clips') or 0):,} & "
        f"{int(train_log.get('max_steps') or 0):,} & {fmt(best_mse, 4)} & {fmt(psnr, 2)} \\\\"
    )


def best_validation_from_log(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    best: tuple[float | None, float | None] = (None, None)
    pattern = re.compile(r"\[val\].*mse=([0-9.]+).*psnr=([0-9.]+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        mse = float(match.group(1))
        psnr = float(match.group(2))
        if best[0] is None or mse < best[0]:
            best = (mse, psnr)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)

    video = sub.add_parser("video")
    video.add_argument("run_dir", type=Path)
    video.add_argument("--label", required=True)

    video_ext = sub.add_parser("video-extended")
    video_ext.add_argument("run_dir", type=Path)
    video_ext.add_argument("--label", required=True)

    diamond = sub.add_parser("diamond")
    diamond.add_argument("run_dir", type=Path)
    diamond.add_argument("--label", required=True)

    args = parser.parse_args()
    if args.kind == "video":
        print(video_row(args.run_dir, label=args.label))
    elif args.kind == "video-extended":
        print(video_extended_row(args.run_dir, label=args.label))
    elif args.kind == "diamond":
        print(diamond_row(args.run_dir, label=args.label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
