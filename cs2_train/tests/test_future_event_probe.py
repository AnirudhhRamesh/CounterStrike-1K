from __future__ import annotations

import json

import numpy as np
import pandas as pd
from counterstrike1k.schema import ACTIONS_DTYPE, BUTTONS, STATE_DTYPE

from cs2_release.core.io import DatasetRoots
from cs2_release.core.windows import build_windows
from cs2_release.future_events.labels import build_future_event_labels
from cs2_release.future_events.train import build_arm_examples, paired_bootstrap


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eval_window_id": "round_a__w000",
                "round_id": "round_a",
                "match_id": "match_a",
                "sample_key": f"round_a__p{pov:02d}",
                "pov_idx": pov,
                "split": "train",
                "map_slug": "dust2",
                "phase_bucket": "mid",
                "start_frame": 2,
                "end_frame": 4,
                "fps": 2.0,
                "frame_tick_stride": 2,
                "end_tick": 108,
            }
            for pov in range(10)
        ]
    )


def test_future_labels_use_only_post_context_interval(tmp_path) -> None:
    for directory in ("actions", "state", "events"):
        (tmp_path / directory).mkdir()
    fire_bit = BUTTONS.index("FIRE")
    reload_bit = BUTTONS.index("RELOAD")
    for pov in range(10):
        key = f"round_a__p{pov:02d}"
        actions = np.zeros(8, dtype=ACTIONS_DTYPE)
        state = np.zeros(8, dtype=STATE_DTYPE)
        state["health"] = 100
        state["active_weapon_id"] = 1
        if pov == 0:
            actions["buttons"][4] |= np.uint16(1 << fire_bit)
        if pov == 1:
            actions["buttons"][5] |= np.uint16(1 << reload_bit)
        if pov == 2:
            state["health"][4:] = 90
        if pov == 3:
            state["active_weapon_id"][5:] = 2
        actions.tofile(tmp_path / "actions" / f"{key}.actions.bin")
        state.tofile(tmp_path / "state" / f"{key}.state.bin")
        (tmp_path / "events" / f"{key}.events.json").write_text(
            json.dumps(
                {
                    "events": [
                        {"type": "bomb_planted", "frame_idx": 3},
                        {"type": "player_death", "frame_idx": 5},
                        {"type": "bomb_exploded", "frame_idx": 6},
                    ]
                }
            ),
            encoding="utf-8",
        )

    labels = build_future_event_labels(
        _windows(),
        roots=DatasetRoots.from_args(root=tmp_path),
        horizon_seconds=1.0,
    )
    assert len(labels) == 1
    row = labels.iloc[0]
    assert (row["future_start_frame"], row["future_end_frame"]) == (4, 6)
    assert row["target_FIRE"] == 1
    assert row["target_RELOAD"] == 1
    assert row["target_DAMAGE"] == 1
    assert row["target_WEAPON_SWITCH"] == 1
    assert row["target_PLAYER_DEATH"] == 1
    assert row["target_BOMB_PLANTED"] == 0
    assert row["target_BOMB_EXPLODED"] == 0


def test_probe_arms_hold_feature_count_and_anchor_contract() -> None:
    label_rows = []
    index_rows = []
    embeddings = []
    row_id = 0
    for match_idx in range(12):
        window_id = f"round_{match_idx:02d}__w000"
        label_rows.append(
            {
                "eval_window_id": window_id,
                "round_id": f"round_{match_idx:02d}",
                "match_id": f"match_{match_idx:02d}",
                "split": "test",
                "map_slug": "dust2",
                "phase_bucket": "mid",
                "target_FIRE": match_idx % 2,
            }
        )
        for pov in range(10):
            index_rows.append(
                {
                    "eval_window_id": window_id,
                    "sample_key": f"round_{match_idx:02d}__p{pov:02d}",
                    "pov_idx": pov,
                }
            )
            embeddings.append([float(row_id), float(match_idx), float(pov)])
            row_id += 1
    labels = pd.DataFrame(label_rows)
    index = pd.DataFrame(index_rows)
    embedding_array = np.asarray(embeddings, dtype=np.float32)

    outputs = {
        arm: build_arm_examples(
            labels,
            index,
            embedding_array,
            ["target_FIRE"],
            split="test",
            arm=arm,
            seed=17,
        )
        for arm in ("single", "synchronized", "shuffled")
    }
    for features, targets, meta in outputs.values():
        assert features.shape == (12, 9)
        assert targets.shape == (12, 1)
        assert len(meta) == 12
    assert set(outputs["single"][2]["feature_rows"]) == {1}
    assert set(outputs["synchronized"][2]["feature_rows"]) == {10}
    assert set(outputs["synchronized"][2]["source_rounds"]) == {1}
    assert set(outputs["shuffled"][2]["feature_rows"]) == {10}
    assert set(outputs["shuffled"][2]["source_rounds"]) == {10}
    assert outputs["single"][2]["anchor_pov"].tolist() == outputs["shuffled"][2][
        "anchor_pov"
    ].tolist()


def test_checkpoint_representation_mode_keeps_identical_synchronized_windows() -> None:
    labels = pd.DataFrame(
        [
            {
                "eval_window_id": "round_a__mira_midpoint",
                "round_id": "round_a",
                "match_id": "match_a",
                "split": "test",
                "map_slug": "dust2",
                "target_FIRE": 1,
            }
        ]
    )
    index = pd.DataFrame(
        [
            {
                "eval_window_id": "round_a__mira_midpoint",
                "sample_key": f"round_a__p{pov:02d}",
                "pov_idx": pov,
            }
            for pov in range(10)
        ]
    )
    embeddings = np.arange(30, dtype=np.float32).reshape(10, 3)

    _, _, meta = build_arm_examples(
        labels,
        index,
        embeddings,
        ["target_FIRE"],
        split="test",
        arm="shuffled",
        seed=17,
        checkpoint_representations=True,
    )

    assert meta["feature_rows"].tolist() == [10]
    assert meta["source_rounds"].tolist() == [1]
    assert meta["source_windows"].iloc[0].split(",") == [
        "round_a__mira_midpoint"
    ] * 10


def test_window_manifest_override_controls_match_atomic_split(tmp_path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "sample_key": f"round_a__p{pov:02d}",
                "round_id": "round_a",
                "match_id": "match_a",
                "round_idx": 1,
                "pov_idx": pov,
                "split": "train",
                "map_slug": "dust2",
                "fps": 32.0,
                "frame_tick_stride": 2,
                "frames": 128,
                "frame0_tick": 100,
                "team_side": "T" if pov < 5 else "CT",
            }
            for pov in range(10)
        ]
    )
    round_index = pd.DataFrame(
        [
            {
                "round_id": "round_a",
                "match_id": "match_a",
                "round_idx": 1,
                "split": "train",
                "map_slug": "dust2",
                "complete_10_pov": True,
                "round_start_tick": 100,
                "round_stop_tick": 356,
                "alive_intersection_start_tick": 100,
                "alive_intersection_end_tick": 356,
                "clip_intersection_start_tick": 100,
                "clip_intersection_end_tick": 356,
            }
        ]
    )
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    round_index.to_parquet(tmp_path / "round_index.parquet", index=False)
    override = manifest.copy()
    override["split"] = "test"
    override_path = tmp_path / "confirmatory.parquet"
    override.to_parquet(override_path, index=False)

    windows = build_windows(
        root=tmp_path,
        subset=None,
        split="test",
        map_slug="dust2",
        window_seconds=1.0,
        windows_per_round=1,
        alive_only=True,
        max_rounds_per_split=None,
        seed=123,
        manifest_path=override_path,
    )
    assert len(windows) == 10
    assert set(windows["split"]) == {"test"}


def test_paired_event_bootstrap_preserves_common_test_windows() -> None:
    rows = []
    for match_idx in range(3):
        for label in (0, 1):
            rows.append(
                {
                    "eval_window_id": f"m{match_idx}_y{label}",
                    "match_id": f"m{match_idx}",
                    "round_id": f"m{match_idx}_r{label}",
                    "target_FIRE_target": label,
                    "target_FIRE_prob": 0.9 if label else 0.1,
                }
            )
    synchronized = pd.DataFrame(rows)
    shuffled = synchronized.copy()
    shuffled["target_FIRE_prob"] = 1.0 - shuffled["target_FIRE_prob"]
    result = paired_bootstrap(
        synchronized,
        shuffled,
        targets=["target_FIRE"],
        left_name="synchronized",
        right_name="shuffled",
        samples=100,
        seed=3,
    )
    assert result["definition"] == "synchronized_minus_shuffled"
    assert result["clusters"] == 3
    assert result["observed"]["macro_ap"] > 0
    assert result["ci95"]["macro_ap"][0] > 0
