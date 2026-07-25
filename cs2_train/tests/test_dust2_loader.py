from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from counterstrike1k.schema import ACTIONS_DTYPE

from cs2_train.src.dataset import CSDataset


class _Frames:
    def __init__(self, indices: list[int]) -> None:
        self.data = torch.stack(
            [torch.full((3, 8, 12), index, dtype=torch.uint8) for index in indices]
        )


class _Decoder:
    def get_frames_at(self, indices: list[int]) -> _Frames:
        return _Frames(indices)


def _write_release(
    root: Path,
    rows: list[dict],
    *,
    manifest_name: str = "confirmatory.parquet",
) -> None:
    (root / "videos" / "360p").mkdir(parents=True)
    (root / "actions").mkdir()
    (root / "state").mkdir()
    (root / "events").mkdir()
    (root / "metadata").mkdir()
    for row in rows:
        key = row["sample_key"]
        (root / "videos" / "360p" / f"{key}.mp4").write_bytes(b"video")
        actions = np.zeros(int(row["frames"]), dtype=ACTIONS_DTYPE)
        actions["tick"] = np.arange(len(actions), dtype=np.uint32)
        actions["delta_pitch"] = 1.0
        actions["delta_yaw"] = 2.0
        actions["buttons"][0] = 1 << 0
        actions["buttons"][1] = 1 << 1
        actions.tofile(root / "actions" / f"{key}.actions.bin")
        (root / "state" / f"{key}.state.bin").write_bytes(b"state")
        death_frame = 10 if int(row["pov_idx"]) == 0 else 20
        (root / "events" / f"{key}.events.json").write_text(
            json.dumps({"events": [{"type": "player_death", "frame_idx": death_frame}]})
        )
        (root / "metadata" / f"{key}.json").write_text("{}")
    pd.DataFrame(rows).to_parquet(root / manifest_name)


def _row(sample_key: str, *, round_id: str, pov_idx: int, frames: int = 41) -> dict:
    return {
        "sample_key": sample_key,
        "match_id": "match",
        "round_id": round_id,
        "round_idx": 1,
        "pov_idx": pov_idx,
        "frames": frames,
        "frame0_tick": 0,
        "fps": 32.0,
        "split": "test",
        "map_slug": "dust2",
    }


def test_32_to_8_fps_actions_are_interval_aggregated(tmp_path: Path) -> None:
    _write_release(tmp_path, [_row("sample", round_id="round", pov_idx=0)])
    dataset = CSDataset(
        tmp_path,
        split="test",
        T=2,
        target_fps=8,
        manifest_name="confirmatory.parquet",
        mode="dict",
        window_mode="sliding",
        resize=None,
    )
    dataset._get_decoder = lambda _path: _Decoder()  # type: ignore[method-assign]

    sample = dataset[0]
    assert sample["start_frame"] == 0
    assert torch.allclose(
        sample["video"][:, 0, 0, 0],
        torch.tensor([-1.0, 4 / 127.5 - 1]),
    )
    assert sample["actions"].shape == (2, 14)
    assert sample["actions"][0, :2].tolist() == [1.0, 1.0]
    assert sample["actions"][0, 12:].tolist() == [4.0, 8.0]
    assert sample["actions"][1, :12].sum().item() == 0


def test_midpoint_and_first_death_share_start_across_povs(tmp_path: Path) -> None:
    rows = [
        _row("pov0", round_id="round", pov_idx=0, frames=41),
        _row("pov1", round_id="round", pov_idx=1, frames=45),
    ]
    _write_release(tmp_path, rows)

    midpoint = CSDataset(
        tmp_path,
        split="test",
        T=2,
        target_fps=8,
        manifest_name="confirmatory.parquet",
        mode="diamond",
        window_mode="midpoint",
    )
    assert midpoint._resolve_window(0)[1] == midpoint._resolve_window(1)[1] == 16

    first_death = CSDataset(
        tmp_path,
        split="test",
        T=2,
        target_fps=8,
        manifest_name="confirmatory.parquet",
        mode="diamond",
        window_mode="first-death",
    )
    # The round-shared anchor is POV 0's first death at frame 10.
    assert first_death._resolve_window(0)[1] == first_death._resolve_window(1)[1] == 6


def test_stride_must_match_target_fps(tmp_path: Path) -> None:
    _write_release(tmp_path, [_row("sample", round_id="round", pov_idx=0)])
    try:
        CSDataset(
            tmp_path,
            split="test",
            T=2,
            target_fps=8,
            stride=1,
            manifest_name="confirmatory.parquet",
        )
    except ValueError as exc:
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("conflicting stride should fail")
