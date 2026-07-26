from __future__ import annotations

import json

import pandas as pd

from cs2_release.retrieval.select_sweep_windows import select_sweep_windows


def test_selects_exact_query_candidate_union(tmp_path) -> None:
    windows = pd.DataFrame([
        {
            "eval_window_id": f"w{idx}",
            "pov_idx": idx,
            "map_slug": "dust2",
            "match_id": "m0",
            "round_idx": 0,
            "window_idx": idx,
        }
        for idx in range(5)
    ])
    pairs_root = tmp_path / "pairs"
    pairs_root.mkdir()
    (pairs_root / "protocol_sweep.json").write_text(
        json.dumps({"status": "pass"}),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "query_eval_window_id": "w1",
            "query_pov_idx": 1,
            "candidate_eval_window_id": "w3",
            "candidate_pov_idx": 3,
        },
        {
            "query_eval_window_id": "w1",
            "query_pov_idx": 1,
            "candidate_eval_window_id": "w4",
            "candidate_pov_idx": 4,
        },
    ]).to_parquet(pairs_root / "cell.parquet", index=False)
    selected, metadata = select_sweep_windows(
        windows,
        pairs_roots=[pairs_root],
    )
    assert list(selected["eval_window_id"]) == ["w1", "w3", "w4"]
    assert metadata["selected_window_rows"] == 3
    assert metadata["referenced_unique_rows"] == 3
