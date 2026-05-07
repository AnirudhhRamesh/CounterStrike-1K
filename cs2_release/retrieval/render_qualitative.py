"""Render qualitative cross-POV retrieval examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from PIL import ImageDraw

from cs2_release.core.io import DatasetRoots, read_parquet, read_video_bytes, write_json
from cs2_release.core.video import decode_sampled_frames, make_frame_grid


ACTION_LABELS = ["FIRE", "RIGHTCLICK", "RELOAD", "JUMP", "DUCK", "WALK", "MOUSE_MOVE"]


def _sample_index(root: Path, resolution: str) -> pd.DataFrame | None:
    for candidate in (root / f"sample_index_{resolution}.parquet", root / "sample_index.parquet"):
        if candidate.exists():
            return read_parquet(candidate)
    return None


def _mid_frame(row: pd.Series) -> int:
    return int((int(row["start_frame"]) + int(row["end_frame"]) - 1) // 2)


def _window_lookup(windows: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    return {
        (str(row["eval_window_id"]), int(row["pov_idx"])): row
        for _, row in windows.iterrows()
    }


def _label_lookup(labels: pd.DataFrame | None) -> dict[tuple[str, int], pd.Series]:
    if labels is None or labels.empty:
        return {}
    required = {"eval_window_id", "pov_idx"}
    if not required.issubset(labels.columns):
        missing = sorted(required - set(labels.columns))
        raise ValueError(f"labels file is missing required columns: {missing}")
    return {
        (str(row["eval_window_id"]), int(row["pov_idx"])): row
        for _, row in labels.iterrows()
    }


def _active_actions(row: pd.Series | None) -> list[str]:
    if row is None:
        return []
    active = []
    for action in ACTION_LABELS:
        col = f"label_{action}"
        if col in row.index and int(row[col]) == 1:
            active.append(action)
    return active


def _action_text(row: pd.Series | None) -> str:
    active = _active_actions(row)
    if not active:
        return "none"
    return ",".join(active[:3]) + ("+" if len(active) > 3 else "")


def _distance_text(row: pd.Series) -> str:
    if "query_candidate_distance" not in row.index or pd.isna(row["query_candidate_distance"]):
        return ""
    return f" d={float(row['query_candidate_distance']):.0f}"


def _policy_text(row: pd.Series) -> str:
    policy = str(row.get("hard_negative_policy", row.get("pair_policy", "")))
    if policy == "same_time_far_location":
        return "neg=same time/far"
    if policy == "same_map_phase_different_round":
        return "neg=same map/phase"
    if policy == "same_match_wrong_round":
        return "neg=same match/wrong round"
    if policy == "same_round_wrong_time":
        return "neg=same round/wrong time"
    return f"policy={policy[:32]}"


def _decode_window_frame(
    *,
    row: pd.Series,
    roots: DatasetRoots,
    sample_index: pd.DataFrame | None,
    resize: tuple[int, int],
) -> object:
    payload = read_video_bytes(
        str(row["sample_key"]),
        roots=roots,
        sample_index=sample_index,
        verify_sha256=False,
    )
    return decode_sampled_frames(payload, [_mid_frame(row)], resize=resize)[0]


def select_examples(predictions: pd.DataFrame, *, successes: int, failures: int) -> list[tuple[str, pd.DataFrame]]:
    examples: list[tuple[str, pd.DataFrame]] = []
    for candidate_set_id, group in predictions.groupby("candidate_set_id", sort=False):
        top = group.sort_values("rank").iloc[0]
        positive = group[group["label"] == 1].sort_values("rank").head(1)
        wrong = group[group["label"] == 0].sort_values("rank").head(1)
        if positive.empty or wrong.empty:
            continue
        is_success = int(top["label"]) == 1
        if is_success and sum(kind == "success" for kind, _ in examples) < successes:
            examples.append(("success", group))
        if not is_success and sum(kind == "failure" for kind, _ in examples) < failures:
            examples.append(("failure", group))
        if (
            sum(kind == "success" for kind, _ in examples) >= successes
            and sum(kind == "failure" for kind, _ in examples) >= failures
        ):
            break
    return examples


def render_examples(
    *,
    windows: pd.DataFrame,
    predictions: pd.DataFrame,
    roots: DatasetRoots,
    labels: pd.DataFrame | None,
    out: Path,
    successes: int,
    failures: int,
    resize: tuple[int, int],
) -> dict:
    lookup = _window_lookup(windows)
    label_rows = _label_lookup(labels)
    sample_index = _sample_index(roots.root, roots.resolution)
    panels = []
    metadata = []
    for kind, group in select_examples(predictions, successes=successes, failures=failures):
        group = group.sort_values("rank")
        top = group.iloc[0]
        positive = group[group["label"] == 1].sort_values("rank").iloc[0]
        wrong = group[group["label"] == 0].sort_values("rank").iloc[0]
        query_row = lookup[(str(top["query_eval_window_id"]), int(top["query_pov_idx"]))]
        pos_row = lookup[(str(positive["candidate_eval_window_id"]), int(positive["candidate_pov_idx"]))]
        wrong_row = lookup[(str(wrong["candidate_eval_window_id"]), int(wrong["candidate_pov_idx"]))]
        query_label = label_rows.get((str(top["query_eval_window_id"]), int(top["query_pov_idx"])))
        pos_label = label_rows.get((str(positive["candidate_eval_window_id"]), int(positive["candidate_pov_idx"])))
        wrong_label = label_rows.get((str(wrong["candidate_eval_window_id"]), int(wrong["candidate_pov_idx"])))
        frames = [
            _decode_window_frame(row=query_row, roots=roots, sample_index=sample_index, resize=resize),
            _decode_window_frame(row=pos_row, roots=roots, sample_index=sample_index, resize=resize),
            _decode_window_frame(row=wrong_row, roots=roots, sample_index=sample_index, resize=resize),
        ]
        labels = [
            f"{kind} q p{int(top['query_pov_idx'])} | {_action_text(query_label)}",
            (
                f"correct p{int(positive['candidate_pov_idx'])} "
                f"r={int(positive['rank'])}{_distance_text(positive)} | {_action_text(pos_label)}"
            ),
            (
                f"wrong p{int(wrong['candidate_pov_idx'])} "
                f"r={int(wrong['rank'])}{_distance_text(wrong)} | {_action_text(wrong_label)}"
            ),
        ]
        grid = make_frame_grid(frames, labels=labels, columns=3, label_height=30)
        panel = Image.fromarray(grid)
        draw = ImageDraw.Draw(panel)
        caption = (
            f"pos=co-located synchronized POV | {_policy_text(top)} | "
            f"{top.get('map_slug', '')}"
        )
        draw.text((6, panel.height - 16), caption, fill=(255, 255, 255))
        panels.append(panel)
        metadata.append({
            "kind": kind,
            "candidate_set_id": str(top["candidate_set_id"]),
            "query_eval_window_id": str(top["query_eval_window_id"]),
            "query_pov_idx": int(top["query_pov_idx"]),
            "correct_eval_window_id": str(positive["candidate_eval_window_id"]),
            "correct_pov_idx": int(positive["candidate_pov_idx"]),
            "positive_rank": int(positive["rank"]),
            "positive_distance": (
                float(positive["query_candidate_distance"])
                if "query_candidate_distance" in positive.index and pd.notna(positive["query_candidate_distance"])
                else None
            ),
            "top_wrong_eval_window_id": str(wrong["candidate_eval_window_id"]),
            "top_wrong_pov_idx": int(wrong["candidate_pov_idx"]),
            "top_wrong_rank": int(wrong["rank"]),
            "top_wrong_distance": (
                float(wrong["query_candidate_distance"])
                if "query_candidate_distance" in wrong.index and pd.notna(wrong["query_candidate_distance"])
                else None
            ),
            "query_actions": _active_actions(query_label),
            "correct_actions": _active_actions(pos_label),
            "top_wrong_actions": _active_actions(wrong_label),
            "map_slug": str(top.get("map_slug", "")),
            "positive_definition": "synchronized co-located POV",
            "policy": str(top.get("hard_negative_policy", top.get("pair_policy", ""))),
        })
    if not panels:
        raise RuntimeError("no qualitative retrieval examples were selected")
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color=(16, 16, 16))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    write_json(out.with_suffix(".json"), {"examples": metadata})
    return {"out": str(out), "examples": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None,
                        help="Optional action_probe_labels.parquet for panel action overlays.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=3)
    parser.add_argument("--failures", type=int, default=3)
    parser.add_argument("--resize", type=int, nargs=2, default=[192, 108], metavar=("W", "H"))
    args = parser.parse_args()

    roots = DatasetRoots.from_args(root=args.root, shard_root=args.shard_root, resolution=args.resolution)
    result = render_examples(
        windows=read_parquet(args.windows),
        predictions=read_parquet(args.predictions),
        roots=roots,
        labels=read_parquet(args.labels) if args.labels is not None else None,
        out=args.out,
        successes=args.successes,
        failures=args.failures,
        resize=(int(args.resize[0]), int(args.resize[1])),
    )
    print(result["out"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
