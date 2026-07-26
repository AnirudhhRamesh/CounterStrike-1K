from __future__ import annotations

import numpy as np
import pandas as pd

from cs2_release.retrieval.evaluate_protocol_sweep import _aggregate_cells, score_pairs


def _index() -> pd.DataFrame:
    rows = []
    for row_id in range(4):
        rows.append({
            "eval_window_id": f"w{row_id}",
            "sample_key": f"s{row_id}",
            "pov_idx": row_id,
            "embedding_row_id": row_id,
        })
    return pd.DataFrame(rows)


def _pairs() -> pd.DataFrame:
    rows = []
    for ordinal, candidate in enumerate(range(1, 4)):
        rows.append({
            "candidate_set_id": "q0",
            "sweep_query_id": "q0",
            "query_window_row_id": 0,
            "candidate_window_row_id": candidate,
            "query_eval_window_id": "w0",
            "candidate_eval_window_id": f"w{candidate}",
            "query_pov_idx": 0,
            "candidate_pov_idx": candidate,
            "query_match_id": "m0",
            "map_slug": "dust2",
            "hard_negative_policy": "same_round_wrong_time",
            "pair_sampling_seed": 17,
            "declared_candidates": 3,
            "candidate_ordinal": ordinal,
            "label": int(candidate == 1),
        })
    return pd.DataFrame(rows)


def test_ties_are_not_broken_by_candidate_row_order() -> None:
    embeddings = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    predictions, summary = score_pairs(_pairs(), _index(), embeddings)
    assert len(predictions) == 3
    row = summary.iloc[0]
    assert row["greater"] == 0
    assert row["tied"] == 2
    assert row["top1"] == 0.5
    assert row["top5"] == 1.0
    assert row["mrr"] == 0.75
    assert row["positive_rank"] == 1.5


def test_spatial_radius_is_retained_as_an_aggregate_dimension() -> None:
    rows = []
    for radius, top1 in [(400.0, 1.0), (800.0, 0.0)]:
        for seed in [17, 29]:
            rows.append({
                "sweep_query_id": "q0",
                "query_match_id": "m0",
                "map_slug": "dust2",
                "hard_negative_policy": "same_location_wrong_time",
                "pair_sampling_seed": seed,
                "declared_candidates": 2,
                "spatial_radius": radius,
                "top1": top1,
                "top5": 1.0,
                "mrr": 0.5 + 0.5 * top1,
                "positive_rank": 2.0 - top1,
                "top1_lift_over_chance": 2.0 * top1 - 1.0,
                "tied": 1,
            })
    aggregate, per_map = _aggregate_cells(
        pd.DataFrame(rows),
        bootstrap_samples=20,
        bootstrap_seed=123,
    )
    assert [row["spatial_radius"] for row in aggregate] == [400.0, 800.0]
    assert [row["top1"] for row in aggregate] == [1.0, 0.0]
    assert [row["spatial_radius"] for row in per_map] == [400.0, 800.0]
