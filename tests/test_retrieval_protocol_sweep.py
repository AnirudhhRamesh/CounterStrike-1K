from __future__ import annotations

import pandas as pd

from cs2_release.retrieval.protocol_sweep import (
    align_sweep_query_cohort,
    build_strict_retrieval_pairs,
    validate_strict_pairs,
)


def _windows(*, windows_per_round: int = 20) -> pd.DataFrame:
    rows = []
    for match_idx in range(3):
        for round_idx in range(2):
            round_id = f"m{match_idx}__r{round_idx}"
            for window_idx in range(windows_per_round):
                for pov_idx in range(10):
                    rows.append({
                        "eval_window_id": f"{round_id}__w{window_idx:03d}",
                        "round_id": round_id,
                        "match_id": f"m{match_idx}",
                        "round_idx": round_idx,
                        "window_idx": window_idx,
                        "sample_key": f"{round_id}__p{pov_idx:02d}",
                        "pov_idx": pov_idx,
                        "split": "test",
                        "map_slug": "dust2",
                        "phase_bucket": "early",
                        "start_tick": window_idx * 64,
                    })
    return pd.DataFrame(rows)


def test_strict_pairs_are_unique_and_candidate_sweep_is_nested() -> None:
    windows = _windows()
    small, _ = build_strict_retrieval_pairs(
        windows,
        split="test",
        candidates_per_query=4,
        max_queries=100,
        negative_policy="same_target_pov_same_round_wrong_time",
        seed=29,
    )
    large, _ = build_strict_retrieval_pairs(
        windows,
        split="test",
        candidates_per_query=16,
        max_queries=100,
        negative_policy="same_target_pov_same_round_wrong_time",
        seed=29,
    )
    validate_strict_pairs(small, candidates_per_query=4)
    validate_strict_pairs(large, candidates_per_query=16)
    assert set(small["candidate_set_id"]) == set(large["candidate_set_id"])
    for candidate_set_id, group in small.groupby("candidate_set_id"):
        small_ids = set(zip(group["candidate_eval_window_id"], group["candidate_pov_idx"]))
        large_group = large[large["candidate_set_id"] == candidate_set_id]
        large_ids = set(zip(large_group["candidate_eval_window_id"], large_group["candidate_pov_idx"]))
        assert small_ids <= large_ids


def test_protocol_never_relaxes_an_insufficient_pool() -> None:
    pairs, diagnostics = build_strict_retrieval_pairs(
        _windows(windows_per_round=3),
        split="test",
        candidates_per_query=8,
        max_queries=20,
        negative_policy="same_target_pov_same_round_wrong_time",
        seed=17,
    )
    assert pairs.empty
    assert diagnostics["skipped_insufficient_unique_negatives"] == 20


def test_all_negative_recipes_can_share_the_exact_query_cohort() -> None:
    windows = _windows()
    tables = {}
    for policy in (
        "same_map_phase_different_round",
        "same_match_wrong_round",
        "same_round_wrong_time",
        "same_target_pov_same_round_wrong_time",
    ):
        table, _ = build_strict_retrieval_pairs(
            windows,
            split="test",
            candidates_per_query=8,
            max_queries=120,
            negative_policy=policy,
            seed=43,
        )
        tables[policy] = table
    aligned, cohort = align_sweep_query_cohort(tables)
    assert len(cohort) == 120
    assert all(table["sweep_query_id"].nunique() == 120 for table in aligned.values())


def test_positive_pov_is_fixed_across_negative_recipes() -> None:
    windows = _windows()
    target_by_policy = {}
    for policy in (
        "same_map_phase_different_round",
        "same_target_pov_same_round_wrong_time",
    ):
        table, _ = build_strict_retrieval_pairs(
            windows,
            split="test",
            candidates_per_query=4,
            max_queries=40,
            negative_policy=policy,
            seed=101,
        )
        target_by_policy[policy] = (
            table[["candidate_set_id", "positive_target_pov_idx"]]
            .drop_duplicates()
            .set_index("candidate_set_id")["positive_target_pov_idx"]
            .to_dict()
        )
    assert target_by_policy[
        "same_map_phase_different_round"
    ] == target_by_policy["same_target_pov_same_round_wrong_time"]
