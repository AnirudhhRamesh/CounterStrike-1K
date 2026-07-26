from __future__ import annotations

import torch

from cs2_train.src.dataset import collate_diamond
from cs2_train.src.diamond import Segment, SegmentId
from cs2_train.src.evaluate_action_sensitivity import (
    build_cross_round_donors,
    select_dataset_indices,
    summarize_valid_rollout,
)
from cs2_train.src.train import apply_action_mode, sequence_derangement


def _segment(index: int, round_id: str) -> Segment:
    return Segment(
        obs=torch.zeros(3, 3, 4, 4),
        act=torch.full((3, 2), float(index)),
        rew=torch.zeros(3),
        end=torch.zeros(3, dtype=torch.uint8),
        trunc=torch.zeros(3, dtype=torch.uint8),
        mask_padding=torch.ones(3, dtype=torch.bool),
        info={
            "sample_key": f"sample-{index}",
            "match_id": "match",
            "round_id": round_id,
        },
        id=SegmentId(episode_id=index, start=0, stop=3),
    )


def test_sequence_shuffle_is_cross_round_and_temporally_coherent() -> None:
    batch = collate_diamond(
        [
            _segment(0, "a"),
            _segment(1, "b"),
            _segment(2, "c"),
            _segment(3, "d"),
        ]
    )
    shuffled = apply_action_mode(
        batch,
        "shuffled",
        generator=torch.Generator().manual_seed(9),
    )
    for target_idx, info in enumerate(shuffled.info):
        donor_idx = int(info["action_source_batch_idx"])
        assert donor_idx != target_idx
        assert batch.info[donor_idx]["round_id"] != batch.info[target_idx]["round_id"]
        assert torch.equal(shuffled.act[target_idx], batch.act[donor_idx])


def test_sequence_shuffle_uses_only_the_dedicated_action_rng() -> None:
    batch = collate_diamond(
        [
            _segment(0, "a"),
            _segment(1, "b"),
            _segment(2, "c"),
            _segment(3, "d"),
        ]
    )
    torch.manual_seed(1234)
    global_rng_before = torch.random.get_rng_state().clone()
    action_generator = torch.Generator().manual_seed(9)
    action_rng_before = action_generator.get_state().clone()

    apply_action_mode(batch, "shuffled", generator=action_generator)

    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert not torch.equal(action_generator.get_state(), action_rng_before)


def test_derangement_rejects_unshufflable_round_batch() -> None:
    try:
        sequence_derangement(3, labels=["same", "same", "same"])
    except ValueError as exc:
        assert "cross-round" in str(exc)
    else:
        raise AssertionError("same-round batch must not be accepted as a control")


def test_eval_donors_keep_pov_and_change_round() -> None:
    infos = [
        {"sample_key": f"{round_id}-{pov}", "round_id": round_id, "pov_idx": pov}
        for round_id in ("r0", "r1", "r2")
        for pov in range(2)
    ]
    donors = build_cross_round_donors(infos, seed=37)
    assert len(set(donors)) == len(donors)
    for index, donor in enumerate(donors):
        assert donor != index
        assert infos[donor]["round_id"] != infos[index]["round_id"]
        assert infos[donor]["pov_idx"] == infos[index]["pov_idx"]


def test_rollout_metrics_mask_post_death_camera_targets() -> None:
    metrics = summarize_valid_rollout(
        [1.0, 2.0, 30.0, 40.0],
        target_source_frames=[96, 100, 104, 108],
        alive_end_frame=101,
    )
    assert metrics["valid_steps"] == [True, True, False, False]
    assert metrics["mse_per_step"] == [1.0, 2.0, None, None]
    assert metrics["valid_count"] == 2
    assert metrics["mean"] == 1.5
    assert metrics["last"] == 2.0


def test_validation_pov_selection_preserves_global_indices() -> None:
    class Dataset:
        def __init__(self) -> None:
            self.rows = [
                {"pov_idx": 0},
                {"pov_idx": 1},
                {"pov_idx": 0},
                {"pov_idx": 1},
                {"pov_idx": 0},
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def _resolve_window(self, index: int):
            return self.rows[index], 0, []

    selected = select_dataset_indices(  # type: ignore[arg-type]
        Dataset(),
        pov_idx=0,
        max_samples=2,
    )
    assert selected == [0, 2]
