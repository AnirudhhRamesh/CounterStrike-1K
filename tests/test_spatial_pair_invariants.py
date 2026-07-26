from __future__ import annotations

from pathlib import Path

import pandas as pd

from cs2_release.retrieval.pairs import spatial
from cs2_release.retrieval.spatial_protocol_sweep import build_spatial_sweep


def _positioned_windows(*, colocated: bool, windows_per_round: int = 2) -> pd.DataFrame:
    rows = []
    for round_idx in range(2):
        for window_idx in range(windows_per_round):
            for pov_idx in range(10):
                rows.append({
                    "eval_window_id": f"r{round_idx}__w{window_idx}",
                    "round_id": f"r{round_idx}",
                    "match_id": "m0",
                    "round_idx": round_idx,
                    "window_idx": window_idx,
                    "sample_key": f"r{round_idx}__p{pov_idx}",
                    "pov_idx": pov_idx,
                    "split": "test",
                    "map_slug": "dust2",
                    "phase_bucket": "early",
                    "team_side": "T" if pov_idx < 5 else "CT",
                    "start_tick": window_idx * 64,
                    "end_tick": window_idx * 64 + 64,
                    "start_frame": window_idx * 32,
                    "end_frame": window_idx * 32 + 32,
                    "pos_x": float(pov_idx * (50 if colocated else 5000)),
                    "pos_y": 0.0,
                    "pos_z": float(pov_idx * (10 if colocated else 500)),
                })
    return pd.DataFrame(rows)


def test_same_time_far_does_not_invent_out_of_radius_positive(monkeypatch) -> None:
    windows = _positioned_windows(colocated=False)
    monkeypatch.setattr(
        spatial,
        "attach_midpoint_positions",
        lambda frame, **_: frame,
    )
    pairs = spatial.build_spatial_retrieval_pairs(
        windows,
        root=Path("."),
        roots=None,
        split="test",
        candidates_per_query=2,
        max_queries=None,
        max_positives=1,
        positive_radius=100.0,
        negative_policy="same_time_far_location",
        negative_min_radius=None,
        negative_location_radius=None,
        same_team_only=False,
        include_z=True,
        max_tick_gap=16,
        seed=123,
    )
    assert pairs.empty


def test_same_time_far_skips_impossible_32_way_sets(monkeypatch) -> None:
    windows = _positioned_windows(colocated=True)
    monkeypatch.setattr(
        spatial,
        "attach_midpoint_positions",
        lambda frame, **_: frame,
    )
    pairs = spatial.build_spatial_retrieval_pairs(
        windows,
        root=Path("."),
        roots=None,
        split="test",
        candidates_per_query=32,
        max_queries=None,
        max_positives=1,
        positive_radius=100.0,
        negative_policy="same_time_far_location",
        negative_min_radius=100.0,
        negative_location_radius=None,
        same_team_only=False,
        include_z=True,
        max_tick_gap=16,
        seed=123,
    )
    assert pairs.empty


def test_radius_sweep_uses_3d_strict_common_cohort() -> None:
    tables, metadata = build_spatial_sweep(
        _positioned_windows(colocated=True, windows_per_round=3),
        root=Path("."),
        split="test",
        radii=[100.0, 200.0],
        seeds=[17, 29],
        candidates_per_query=4,
        max_queries=20,
        query_seed=123,
    )
    assert metadata["status"] == "pass"
    assert metadata["include_z"] is True
    assert metadata["aligned_queries"] > 0
    assert len(tables) == 4
    for table in tables.values():
        assert table["candidate_set_id"].nunique() == metadata["aligned_queries"]
        assert table.groupby("candidate_set_id").size().eq(4).all()
        assert not table.groupby("candidate_set_id")["candidate_window_row_id"].apply(
            lambda values: values.duplicated().any()
        ).any()
