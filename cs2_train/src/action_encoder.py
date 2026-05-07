"""CS2 -> DIAMOND-CSGO action encoder.

DIAMOND-CSGO encodes actions as a 51-dim multi-hot vector:
    [11 keys] [l_click] [r_click] [23 mouse_x onehot] [15 mouse_y onehot]

Keys (in order): w, a, s, d, space, ctrl, shift, 1, 2, 3, r

Our locked CS2-WM schema stores per-frame actions as 12 binary buttons + 2
continuous mouse deltas:
    buttons: FORWARD, BACK, LEFT, RIGHT, JUMP, DUCK, WALK, FIRE, RIGHTCLICK,
             RELOAD, INSPECT, USE
    mouse:   delta_pitch, delta_yaw

Mapping:
    FORWARD     -> key 'w'           (idx 0)
    LEFT        -> key 'a'           (idx 1)
    BACK        -> key 's'           (idx 2)
    RIGHT       -> key 'd'           (idx 3)
    JUMP        -> key 'space'       (idx 4)
    DUCK        -> key 'ctrl'        (idx 5)
    WALK        -> key 'shift'       (idx 6)
    RELOAD      -> key 'r'           (idx 10)
    FIRE        -> l_click           (idx 11)
    RIGHTCLICK  -> r_click           (idx 12)
    INSPECT/USE -> unmapped (zero)   no DIAMOND-CSGO equivalent
    delta_yaw   -> mouse_x bucket    (horizontal look)
    delta_pitch -> mouse_y bucket    (vertical look)

Weapon slots 1/2/3 remain zero because CS2-WM records active weapon as state,
not weapon-switch keypresses. Keeping the 51-dim DIAMOND action space unchanged
is intentional baseline fidelity; this file is only an input adapter.
"""

from __future__ import annotations

import numpy as np
import torch

# DIAMOND-CSGO discrete mouse buckets — must match the released checkpoint's
# action space exactly so we can plug in their model unchanged.
MOUSE_X_POSSIBLES = [
    -1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2,
    0,
    2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000,
]  # 23 buckets

MOUSE_Y_POSSIBLES = [
    -200, -100, -50, -20, -10, -4, -2,
    0,
    2, 4, 10, 20, 50, 100, 200,
]  # 15 buckets — the upstream code labels this 16 due to an off-by-one but
   # actually has 15 entries; we use what the keymap provides.

N_KEYS = 11
N_CLICKS = 2
N_MOUSE_X = len(MOUSE_X_POSSIBLES)
N_MOUSE_Y = len(MOUSE_Y_POSSIBLES)
NUM_ACTIONS = N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y  # 11 + 2 + 23 + 15 = 51

CS2_BUTTON_COLS = [
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
CS2_MOUSE_COLS = ["delta_pitch", "delta_yaw"]

# CS2 button index -> DIAMOND key index (None = drop)
_BUTTON_TO_KEY: dict[int, int] = {
    0: 0,    # FORWARD -> w
    2: 1,    # LEFT    -> a
    1: 2,    # BACK    -> s
    3: 3,    # RIGHT   -> d
    4: 4,    # JUMP    -> space
    5: 5,    # DUCK    -> ctrl/crouch
    6: 6,    # WALK    -> shift
    9: 10,   # RELOAD  -> r
}
# CS2 button index -> click slot (0=l_click, 1=r_click)
_BUTTON_TO_CLICK: dict[int, int] = {
    7: 0,  # FIRE       -> l_click
    8: 1,  # RIGHTCLICK -> r_click
}


def _bucketize(values: np.ndarray, buckets: list[int]) -> np.ndarray:
    """Snap each value to the index of its nearest bucket center."""
    bucket_arr = np.asarray(buckets, dtype=np.float32)
    # |v - b| over (..., len(buckets))
    diff = np.abs(values[..., None] - bucket_arr)
    return diff.argmin(axis=-1).astype(np.int64)


def encode_cs2_actions(buttons: np.ndarray, mouse: np.ndarray) -> np.ndarray:
    """Encode CS2 frames-of-actions to DIAMOND multi-hot.

    Args:
        buttons: (T, 12) int/float, values in {0, 1}.
        mouse:   (T, 2) float, [delta_pitch, delta_yaw] per frame.

    Returns:
        (T, 51) float32 multi-hot.
    """
    assert buttons.shape[1] == 12, f"buttons must be (T,12), got {buttons.shape}"
    assert mouse.shape[1] == 2, f"mouse must be (T,2), got {mouse.shape}"
    T = buttons.shape[0]

    out = np.zeros((T, NUM_ACTIONS), dtype=np.float32)

    # keys
    for cs2_idx, key_idx in _BUTTON_TO_KEY.items():
        out[:, key_idx] = buttons[:, cs2_idx].astype(np.float32)

    # clicks
    for cs2_idx, slot in _BUTTON_TO_CLICK.items():
        out[:, N_KEYS + slot] = buttons[:, cs2_idx].astype(np.float32)

    # mouse x onehot from delta_yaw
    yaw_bucket = _bucketize(mouse[:, 1], MOUSE_X_POSSIBLES)
    out[np.arange(T), N_KEYS + N_CLICKS + yaw_bucket] = 1.0

    # mouse y onehot from delta_pitch
    pitch_bucket = _bucketize(mouse[:, 0], MOUSE_Y_POSSIBLES)
    out[np.arange(T), N_KEYS + N_CLICKS + N_MOUSE_X + pitch_bucket] = 1.0

    return out


def encode_cs2_actions_torch(buttons: torch.Tensor, mouse: torch.Tensor) -> torch.Tensor:
    """Torch wrapper around the numpy encoder. Returns float32 (T, 51)."""
    enc = encode_cs2_actions(buttons.detach().cpu().numpy(), mouse.detach().cpu().numpy())
    return torch.from_numpy(enc)
