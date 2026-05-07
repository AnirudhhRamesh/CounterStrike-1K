"""Build per-window action labels from public match state sidecars.

Two backends are supported:

* ``match_states`` (default): read pre-aggregated ``match_states/match_<id>.parquet``
  files from the dataset root.
* ``actions_bin``: parse per-sample ``<sample_key>.actions.bin`` records directly
  from the public WebDataset shards (or unsharded sample layout). This is the
  release-only path — the public Hugging Face mirrors do not redistribute the
  match_states aggregation parquets.

The script auto-falls back from ``match_states`` to ``actions_bin`` when the
match_states parquet is missing. Both backends produce identical output rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_release.core.io import (
    DatasetRoots,
    dataframe_sha256,
    git_commit,
    read_member_bytes,
    read_parquet,
    read_release_parquet,
    write_json,
)


BUTTON_COLUMNS = [
    "FORWARD",
    "BACK",
    "LEFT",
    "RIGHT",
    "JUMP",
    "DUCK",
    "WALK",
    "FIRE",
    "RIGHTCLICK",
    "RELOAD",
    "INSPECT",
    "USE",
]

# Schema of <sample_key>.actions.bin v9: 14-byte packed records with little-endian fields.
_ACTIONS_BIN_DTYPE = np.dtype([
    ("tick", "<u4"),
    ("delta_pitch", "<f4"),
    ("delta_yaw", "<f4"),
    ("buttons", "<u2"),
])


def _state_path(match_id: str) -> Path:
    return Path("match_states") / f"match_{match_id}.parquet"


def _row_for_window(
    *,
    window: pd.Series,
    button_active: np.ndarray,
    delta_pitch: np.ndarray,
    delta_yaw: np.ndarray,
    n_rows: int,
    min_positive_frames: int,
    mouse_mean_abs_threshold: float,
) -> dict:
    out = {
        "eval_window_id": str(window["eval_window_id"]),
        "sample_key": str(window["sample_key"]),
        "pov_idx": int(window["pov_idx"]),
        "split": str(window["split"]),
        "map_slug": str(window["map_slug"]),
        "round_id": str(window["round_id"]),
        "match_id": str(window["match_id"]),
        "start_tick": int(window["start_tick"]),
        "end_tick": int(window["end_tick"]),
        "state_rows": int(n_rows),
    }
    for idx, button in enumerate(BUTTON_COLUMNS):
        values = button_active[:, idx]
        active_frames = int(values.sum())
        out[f"frac_{button}"] = float(values.mean()) if values.size else 0.0
        out[f"label_{button}"] = int(active_frames >= min_positive_frames)
    mean_abs_pitch = float(np.abs(delta_pitch).mean()) if delta_pitch.size else 0.0
    mean_abs_yaw = float(np.abs(delta_yaw).mean()) if delta_yaw.size else 0.0
    out["mean_abs_delta_pitch"] = mean_abs_pitch
    out["mean_abs_delta_yaw"] = mean_abs_yaw
    out["label_MOUSE_MOVE"] = int(max(mean_abs_pitch, mean_abs_yaw) >= mouse_mean_abs_threshold)
    return out


def build_action_probe_labels(
    windows: pd.DataFrame,
    *,
    root: Path,
    min_positive_frames: int,
    mouse_mean_abs_threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for match_id, match_windows in windows.groupby("match_id", sort=False):
        try:
            state = read_release_parquet(root, _state_path(str(match_id)))
        except FileNotFoundError:
            continue
        needed = {"tick", "pov_idx", "delta_pitch", "delta_yaw", *BUTTON_COLUMNS}
        if not needed.issubset(state.columns):
            continue
        state = state[["tick", "pov_idx", "delta_pitch", "delta_yaw", *BUTTON_COLUMNS]].copy()
        state["tick"] = state["tick"].astype(np.int64)
        state["pov_idx"] = state["pov_idx"].astype(np.int16)
        by_pov = {
            int(pov): group.sort_values("tick").reset_index(drop=True)
            for pov, group in state.groupby("pov_idx", sort=False)
        }
        for _, window in match_windows.iterrows():
            pov_state = by_pov.get(int(window["pov_idx"]))
            if pov_state is None or pov_state.empty:
                continue
            ticks = pov_state["tick"].to_numpy(dtype=np.int64)
            start = int(window["start_tick"])
            end = int(window["end_tick"])
            lo = int(np.searchsorted(ticks, start, side="left"))
            hi = int(np.searchsorted(ticks, end, side="left"))
            if hi <= lo:
                continue
            segment = pov_state.iloc[lo:hi]
            button_active = np.stack(
                [segment[btn].astype(bool).to_numpy() for btn in BUTTON_COLUMNS],
                axis=1,
            )
            rows.append(_row_for_window(
                window=window,
                button_active=button_active,
                delta_pitch=segment["delta_pitch"].to_numpy(dtype=np.float32),
                delta_yaw=segment["delta_yaw"].to_numpy(dtype=np.float32),
                n_rows=int(len(segment)),
                min_positive_frames=min_positive_frames,
                mouse_mean_abs_threshold=mouse_mean_abs_threshold,
            ))
    return pd.DataFrame(rows)


def build_action_probe_labels_from_actions_bin(
    windows: pd.DataFrame,
    *,
    roots: DatasetRoots,
    min_positive_frames: int,
    mouse_mean_abs_threshold: float,
) -> pd.DataFrame:
    """Build per-window labels by reading per-sample ``actions.bin`` records.

    Used when the canonical ``match_states/`` parquets are not redistributed
    with the release shards. Each window is sliced by ``[start_frame, end_frame)``
    against the per-frame action records for that POV's sample.
    """

    sample_index = None
    for candidate in (
        roots.root / f"sample_index_{roots.resolution}.parquet",
        roots.root / "sample_index.parquet",
    ):
        if candidate.exists():
            sample_index = read_parquet(candidate)
            break

    rows: list[dict] = []
    for _, window in windows.iterrows():
        sample_key = str(window["sample_key"])
        try:
            payload = read_member_bytes(
                sample_key, "actions.bin", roots=roots, sample_index=sample_index,
            )
        except (FileNotFoundError, ValueError):
            continue
        n_records = len(payload) // _ACTIONS_BIN_DTYPE.itemsize
        if n_records <= 0:
            continue
        arr = np.frombuffer(payload, dtype=_ACTIONS_BIN_DTYPE, count=n_records)
        start_frame = int(window["start_frame"])
        end_frame = int(window["end_frame"])
        if start_frame < 0 or end_frame > n_records or end_frame <= start_frame:
            continue
        seg = arr[start_frame:end_frame]
        buttons = seg["buttons"].astype(np.uint16, copy=False)
        button_active = np.stack(
            [(buttons >> i) & np.uint16(1) for i in range(len(BUTTON_COLUMNS))],
            axis=1,
        ).astype(bool)
        rows.append(_row_for_window(
            window=window,
            button_active=button_active,
            delta_pitch=seg["delta_pitch"].astype(np.float32, copy=False),
            delta_yaw=seg["delta_yaw"].astype(np.float32, copy=False),
            n_rows=int(end_frame - start_frame),
            min_positive_frames=min_positive_frames,
            mouse_mean_abs_threshold=mouse_mean_abs_threshold,
        ))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None,
                        help="WebDataset shard root for the actions_bin backend.")
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p",
                        help="Resolution of the sample_index used by the actions_bin backend.")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--min-positive-frames", type=int, default=1)
    parser.add_argument("--mouse-mean-abs-threshold", type=float, default=0.35)
    parser.add_argument("--backend", choices=["auto", "match_states", "actions_bin"],
                        default="auto",
                        help="Where to source labels from. 'auto' tries match_states then actions_bin.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = read_parquet(args.windows)
    labels: pd.DataFrame = pd.DataFrame()
    backend_used: str = args.backend
    if args.backend in {"auto", "match_states"}:
        labels = build_action_probe_labels(
            windows,
            root=args.root,
            min_positive_frames=args.min_positive_frames,
            mouse_mean_abs_threshold=args.mouse_mean_abs_threshold,
        )
        backend_used = "match_states"
        if labels.empty and args.backend == "match_states":
            raise RuntimeError(
                "match_states backend produced no labels; pass --backend actions_bin to "
                "use the per-sample actions.bin fallback."
            )
    if labels.empty and args.backend in {"auto", "actions_bin"}:
        roots = DatasetRoots.from_args(
            root=args.root, shard_root=args.shard_root, resolution=args.resolution,
        )
        labels = build_action_probe_labels_from_actions_bin(
            windows,
            roots=roots,
            min_positive_frames=args.min_positive_frames,
            mouse_mean_abs_threshold=args.mouse_mean_abs_threshold,
        )
        backend_used = "actions_bin"
    if labels.empty:
        raise RuntimeError("no action probe labels were produced")
    out_path = args.out / "action_probe_labels.parquet" if args.out.suffix == "" else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_path, index=False)
    label_cols = [c for c in labels.columns if c.startswith("label_")]
    prevalence = {c: float(labels[c].mean()) for c in label_cols}
    write_json(out_path.with_suffix(".metadata.json"), {
        "rows": int(len(labels)),
        "eval_windows": int(labels["eval_window_id"].nunique()),
        "label_prevalence": prevalence,
        "min_positive_frames": args.min_positive_frames,
        "mouse_mean_abs_threshold": args.mouse_mean_abs_threshold,
        "backend": backend_used,
        "labels_sha256": dataframe_sha256(labels),
        "git_commit": git_commit(),
    })
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
