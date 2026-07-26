from __future__ import annotations

import numpy as np

from cs2_release.encoders.registry import RGBHistEncoder


def test_grouped_encoding_preserves_window_semantics() -> None:
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, size=(9, 16, 16, 3), dtype=np.uint8)
    groups = [[0, 2, 4, 6], [1, 3, 5, 7], [2, 5, 8]]
    encoder = RGBHistEncoder()
    expected = np.stack(
        [encoder.encode(frames[np.asarray(group)]) for group in groups],
        axis=0,
    )
    actual = encoder.encode_many(frames, groups, batch_size=32)
    np.testing.assert_array_equal(actual, expected)
