"""Frozen-video features and a temporal CS2 action-recoverability probe.

MIRA's Action Recoverability Ratio (ARR) evaluates whether the action used to
condition a rollout can be decoded from the generated video.  DINOv3 weights
are gated, so this public baseline uses a commit-pinned DINOv2-B/14 backbone
and records the downloaded weight hash.  Only the approximately four-million
parameter temporal head is trained.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DINOV2_REPOSITORY = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_MODEL = "dinov2_vitb14"
DINOV2_WEIGHT_FILENAME = "dinov2_vitb14_pretrain.pth"
DINOV2_EXPECTED_WEIGHT_SHA256 = (
    "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
)
DINOV2_FEATURE_DIM = 768
PROBE_FRAMES = 5
PROBE_TRANSITIONS = 4
MOUSE_DIRECTION_THRESHOLD_DEGREES = 1.0
COMMON_BUTTON_NAMES = (
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
)
ACTION_LABEL_NAMES = (
    *COMMON_BUTTON_NAMES,
    "PITCH_NEG",
    "PITCH_POS",
    "YAW_NEG",
    "YAW_POS",
)


def binary_average_precision_tie_aware(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Return threshold-based binary AP with exact score ties grouped.

    Grouping equal scores matches the precision-recall threshold definition
    used by standard metric libraries.  It is important for ARR because
    quantized probe outputs and cluster-bootstrap multiplicities otherwise make
    AP depend on arbitrary row order.
    """

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be equal-length vectors")
    if sample_weight is None:
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != labels.shape:
            raise ValueError("sample weights must match labels")
    if not np.isfinite(labels).all():
        raise ValueError("labels must be finite")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample weights must be finite and non-negative")

    valid = np.isfinite(scores) & (weights > 0)
    if not valid.any():
        return float("nan")
    labels = labels[valid].astype(np.float64, copy=False)
    scores = scores[valid]
    weights = weights[valid]
    total_positive = float((labels * weights).sum())
    if total_positive <= 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    positive_weights = labels[order] * weights[order]
    weights = weights[order]
    group_ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    cumulative_positive = np.cumsum(positive_weights)
    cumulative_weight = np.cumsum(weights)
    positive_at_threshold = cumulative_positive[group_ends]
    group_positive = np.diff(np.r_[0.0, positive_at_threshold])
    precision = positive_at_threshold / cumulative_weight[group_ends]
    return float((group_positive * precision).sum() / total_positive)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actions_to_labels(
    actions: torch.Tensor,
    *,
    mouse_threshold_degrees: float = MOUSE_DIRECTION_THRESHOLD_DEGREES,
) -> torch.Tensor:
    """Map ``[..., 4, 14]`` canonical CS2 actions to 14 binary labels."""

    if actions.shape[-2:] != (PROBE_TRANSITIONS, 14):
        raise ValueError(
            f"expected [...,{PROBE_TRANSITIONS},14] actions, got {actions.shape}"
        )
    buttons = actions[..., : len(COMMON_BUTTON_NAMES)].amax(dim=-2) > 0
    mouse = actions[..., 12:14].sum(dim=-2)
    directional = torch.stack(
        (
            mouse[..., 0] < -mouse_threshold_degrees,
            mouse[..., 0] > mouse_threshold_degrees,
            mouse[..., 1] < -mouse_threshold_degrees,
            mouse[..., 1] > mouse_threshold_degrees,
        ),
        dim=-1,
    )
    return torch.cat((buttons, directional), dim=-1).to(torch.float32)


def actions_to_labels_numpy(
    actions: np.ndarray,
    *,
    mouse_threshold_degrees: float = MOUSE_DIRECTION_THRESHOLD_DEGREES,
) -> np.ndarray:
    tensor = torch.from_numpy(np.array(actions, dtype=np.float32, copy=True))
    return actions_to_labels(
        tensor,
        mouse_threshold_degrees=mouse_threshold_degrees,
    ).numpy()


def split_probe_segments(
    frame_features: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split nine frames/eight transitions into two fixed 0.5-second probes."""

    if frame_features.ndim != 3 or frame_features.shape[1] != 9:
        raise ValueError(f"expected [B,9,D] frame features, got {frame_features.shape}")
    if actions.shape != (frame_features.shape[0], 8, 14):
        raise ValueError(f"expected [B,8,14] actions, got {actions.shape}")
    features = torch.stack(
        (frame_features[:, 0:5], frame_features[:, 4:9]),
        dim=1,
    )
    action_segments = torch.stack((actions[:, 0:4], actions[:, 4:8]), dim=1)
    labels = actions_to_labels(action_segments)
    return features, labels


class TemporalActionProbe(nn.Module):
    """Small temporal transformer over frozen per-frame visual features."""

    def __init__(
        self,
        *,
        input_dim: int = DINOV2_FEATURE_DIM,
        hidden_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_labels: int = len(ACTION_LABEL_NAMES),
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.num_labels = int(num_labels)
        self.input_norm = nn.LayerNorm(2 * self.input_dim)
        self.input_projection = nn.Linear(2 * self.input_dim, self.hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, PROBE_FRAMES, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
            enable_nested_tensor=False,
        )
        self.pool_query = nn.Parameter(torch.empty(self.hidden_dim))
        self.output = nn.Linear(self.hidden_dim, self.num_labels)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if (
            frame_features.ndim != 3
            or frame_features.shape[1] != PROBE_FRAMES
            or frame_features.shape[2] != self.input_dim
        ):
            raise ValueError(
                f"expected [B,{PROBE_FRAMES},{self.input_dim}], "
                f"got {frame_features.shape}"
            )
        delta = frame_features - frame_features[:, :1]
        tokens = torch.cat((frame_features, delta), dim=-1)
        tokens = self.input_projection(self.input_norm(tokens)) + self.position
        tokens = self.temporal(tokens)
        attention = torch.softmax(
            (tokens * self.pool_query).sum(dim=-1) / math.sqrt(self.hidden_dim),
            dim=1,
        )
        pooled = (tokens * attention.unsqueeze(-1)).sum(dim=1)
        return self.output(pooled)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "num_labels": self.num_labels,
        }


class DinoV2FrameEncoder:
    """Commit-pinned DINOv2-B/14 frame encoder with aspect-preserving padding."""

    def __init__(
        self,
        *,
        device: torch.device,
        input_size: int = 224,
        frame_batch_size: int = 128,
    ) -> None:
        self.device = device
        self.input_size = int(input_size)
        self.frame_batch_size = int(frame_batch_size)
        if self.input_size <= 0 or self.frame_batch_size <= 0:
            raise ValueError("DINO input and batch sizes must be positive")
        self.model = (
            torch.hub.load(
                DINOV2_REPOSITORY,
                DINOV2_MODEL,
                trust_repo=True,
                skip_validation=True,
                verbose=False,
            )
            .to(self.device)
            .eval()
        )
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected [N,3,H,W] frames, got {frames.shape}")
        if frames.dtype == torch.uint8:
            frames = frames.float().div(255)
        else:
            frames = frames.float()
            if float(frames.min()) < -0.01:
                frames = frames.add(1).div(2)
        frames = frames.clamp(0, 1)
        height, width = frames.shape[-2:]
        scale = min(self.input_size / height, self.input_size / width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        resized = F.interpolate(
            frames,
            (resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = self.mean.to(frames.device)
        canvas = mean.expand(
            len(frames),
            3,
            self.input_size,
            self.input_size,
        ).clone()
        top = (self.input_size - resized_height) // 2
        left = (self.input_size - resized_width) // 2
        canvas[
            :,
            :,
            top : top + resized_height,
            left : left + resized_width,
        ] = resized
        return (canvas - mean) / self.std.to(frames.device)

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Return normalized ``[B,T,768]`` features for ``[B,T,3,H,W]``."""

        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W] video, got {video.shape}")
        batch, time = video.shape[:2]
        flat = video.reshape(-1, *video.shape[2:])
        outputs = []
        for start in range(0, len(flat), self.frame_batch_size):
            inputs = self._preprocess(flat[start : start + self.frame_batch_size]).to(
                self.device
            )
            features = self.model(inputs)
            if isinstance(features, dict):
                features = features.get("x_norm_clstoken")
            if not isinstance(features, torch.Tensor):
                raise TypeError("DINOv2 did not return a tensor class token")
            outputs.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(outputs).reshape(batch, time, DINOV2_FEATURE_DIM)

    def provenance(self) -> dict:
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / DINOV2_WEIGHT_FILENAME
        if not checkpoint.is_file():
            model_url = getattr(self.model, "url", None)
            if model_url:
                checkpoint = (
                    Path(torch.hub.get_dir())
                    / "checkpoints"
                    / Path(urlparse(model_url).path).name
                )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "cannot locate the loaded DINOv2 checkpoint for hashing"
            )
        observed_sha256 = sha256_file(checkpoint)
        if observed_sha256 != DINOV2_EXPECTED_WEIGHT_SHA256:
            raise ValueError(
                "DINOv2 checkpoint SHA-256 differs from the frozen ARR contract: "
                f"{observed_sha256}"
            )
        return {
            "implementation": "facebookresearch/dinov2 torch.hub",
            "repository": DINOV2_REPOSITORY,
            "model": DINOV2_MODEL,
            "feature_dim": DINOV2_FEATURE_DIM,
            "weight_filename": checkpoint.name,
            "weights_sha256": observed_sha256,
            "input_size": self.input_size,
            "preprocess": (
                "aspect-preserving bilinear antialias resize; "
                "ImageNet-mean letterbox; ImageNet normalization"
            ),
            "torch_version": torch.__version__,
        }
