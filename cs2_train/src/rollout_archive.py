"""Atomic, model-agnostic rollout archives for downstream dynamics metrics.

The confirmatory evaluator is substantially more expensive than perceptual or
probe-based scoring.  This module retains its paired predictions in standard
NumPy arrays so new metrics can be computed without sampling the world model
again.  ``metadata.json`` is written last and is the completion marker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ARRAY_SPECS = {
    "predictions_uint8": "eval_seed,action_mode,sample,time,height,width,channel",
    "ground_truth_uint8": "sample,time,height,width,channel",
    "context_last_uint8": "sample,height,width,channel",
    "conditioning_actions_model_float32": "action_mode,sample,time,model_action",
    "conditioning_actions_cs2_float32": "action_mode,sample,time,cs2_action",
    "valid_steps_bool": "sample,time",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Convert ``[..., C, H, W]`` frames in ``[-1, 1]`` to uint8 NHWC."""

    if frames.ndim < 4 or frames.shape[-3] != 3:
        raise ValueError(f"expected [..., 3, H, W] frames, got {tuple(frames.shape)}")
    leading = frames.shape[:-3]
    height, width = frames.shape[-2:]
    flat = (
        frames.detach()
        .float()
        .cpu()
        .clamp(-1, 1)
        .add(1)
        .div(2)
        .mul(255)
        .round()
        .byte()
        .reshape(-1, 3, height, width)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    return flat.reshape(*leading, height, width, 3)


class RolloutArchiveWriter:
    """Incrementally write one paired evaluation into atomic ``.npy`` arrays."""

    def __init__(
        self,
        out_dir: Path,
        *,
        num_samples: int,
        eval_seeds: list[int],
        action_modes: list[str],
        rollout_steps: int,
        height: int,
        width: int,
        num_model_actions: int,
        num_cs2_actions: int,
    ) -> None:
        if min(num_samples, len(eval_seeds), len(action_modes), rollout_steps) <= 0:
            raise ValueError("rollout archive dimensions must be positive")
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.eval_seeds = list(eval_seeds)
        self.action_modes = list(action_modes)
        self.num_samples = int(num_samples)
        self.rollout_steps = int(rollout_steps)
        self.height = int(height)
        self.width = int(width)
        self.num_model_actions = int(num_model_actions)
        self.num_cs2_actions = int(num_cs2_actions)
        self._final_paths = {
            name: self.out_dir / f"{name}.npy" for name in ARRAY_SPECS
        }
        self._tmp_paths = {
            name: self.out_dir / f".{name}.tmp.npy" for name in ARRAY_SPECS
        }
        completion = self.out_dir / "metadata.json"
        occupied = [
            path
            for path in [completion, *self._final_paths.values(), *self._tmp_paths.values()]
            if path.exists()
        ]
        if occupied:
            raise FileExistsError(
                f"rollout archive destination is not empty: {occupied[0]}"
            )

        s = len(self.eval_seeds)
        m = len(self.action_modes)
        n = self.num_samples
        t = self.rollout_steps
        h, w = self.height, self.width
        self.predictions = np.lib.format.open_memmap(
            self._tmp_paths["predictions_uint8"],
            mode="w+",
            dtype=np.uint8,
            shape=(s, m, n, t, h, w, 3),
        )
        self.ground_truth = np.lib.format.open_memmap(
            self._tmp_paths["ground_truth_uint8"],
            mode="w+",
            dtype=np.uint8,
            shape=(n, t, h, w, 3),
        )
        self.context_last = np.lib.format.open_memmap(
            self._tmp_paths["context_last_uint8"],
            mode="w+",
            dtype=np.uint8,
            shape=(n, h, w, 3),
        )
        self.conditioning_actions_model = np.lib.format.open_memmap(
            self._tmp_paths["conditioning_actions_model_float32"],
            mode="w+",
            dtype=np.float32,
            shape=(m, n, t, self.num_model_actions),
        )
        self.conditioning_actions_cs2 = np.lib.format.open_memmap(
            self._tmp_paths["conditioning_actions_cs2_float32"],
            mode="w+",
            dtype=np.float32,
            shape=(m, n, t, self.num_cs2_actions),
        )
        self.valid_steps = np.lib.format.open_memmap(
            self._tmp_paths["valid_steps_bool"],
            mode="w+",
            dtype=np.bool_,
            shape=(n, t),
        )
        self._base_written = np.zeros(n, dtype=np.bool_)
        self._actions_written = np.zeros((m, n), dtype=np.bool_)
        self._predictions_written = np.zeros((s, m, n), dtype=np.bool_)

    def write_batch(
        self,
        *,
        seed_index: int,
        mode_index: int,
        sample_positions: list[int],
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        context_last: torch.Tensor,
        conditioning_actions_model: torch.Tensor,
        conditioning_actions_cs2: torch.Tensor,
        valid_steps: np.ndarray,
    ) -> None:
        if not 0 <= seed_index < len(self.eval_seeds):
            raise IndexError("evaluation-seed index is outside the archive")
        if not 0 <= mode_index < len(self.action_modes):
            raise IndexError("action-mode index is outside the archive")
        positions = np.asarray(sample_positions, dtype=np.int64)
        if (
            positions.ndim != 1
            or positions.size == 0
            or len(np.unique(positions)) != len(positions)
        ):
            raise ValueError("sample positions must be a unique one-dimensional list")
        if np.any(positions < 0) or np.any(positions >= self.num_samples):
            raise IndexError("sample position is outside the archive")
        batch_size = len(positions)
        expected_frames = (
            batch_size,
            self.rollout_steps,
            3,
            self.height,
            self.width,
        )
        if tuple(predictions.shape) != expected_frames:
            raise ValueError(
                f"prediction shape {tuple(predictions.shape)} != {expected_frames}"
            )
        if tuple(ground_truth.shape) != expected_frames:
            raise ValueError(
                f"ground-truth shape {tuple(ground_truth.shape)} != {expected_frames}"
            )
        expected_context = (batch_size, 3, self.height, self.width)
        if tuple(context_last.shape) != expected_context:
            raise ValueError(
                f"context shape {tuple(context_last.shape)} != {expected_context}"
            )
        expected_model_actions = (
            batch_size,
            self.rollout_steps,
            self.num_model_actions,
        )
        if tuple(conditioning_actions_model.shape) != expected_model_actions:
            raise ValueError(
                f"model-action shape {tuple(conditioning_actions_model.shape)} != "
                f"{expected_model_actions}"
            )
        expected_cs2_actions = (
            batch_size,
            self.rollout_steps,
            self.num_cs2_actions,
        )
        if tuple(conditioning_actions_cs2.shape) != expected_cs2_actions:
            raise ValueError(
                f"CS2-action shape {tuple(conditioning_actions_cs2.shape)} != "
                f"{expected_cs2_actions}"
            )
        valid = np.asarray(valid_steps, dtype=np.bool_)
        if valid.shape != (batch_size, self.rollout_steps):
            raise ValueError(
                f"valid-step shape {valid.shape} != "
                f"{(batch_size, self.rollout_steps)}"
            )
        if self._predictions_written[seed_index, mode_index, positions].any():
            raise ValueError("attempted to overwrite archived predictions")

        predictions_uint8 = frames_to_uint8(predictions)
        ground_truth_uint8 = frames_to_uint8(ground_truth)
        context_last_uint8 = frames_to_uint8(context_last)
        model_actions_float32 = (
            conditioning_actions_model.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        cs2_actions_float32 = (
            conditioning_actions_cs2.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        self.predictions[seed_index, mode_index, positions] = predictions_uint8
        self._predictions_written[seed_index, mode_index, positions] = True

        base_written = self._base_written[positions]
        if base_written.any():
            written_positions = positions[base_written]
            if (
                not np.array_equal(
                    self.ground_truth[written_positions],
                    ground_truth_uint8[base_written],
                )
                or not np.array_equal(
                    self.context_last[written_positions],
                    context_last_uint8[base_written],
                )
                or not np.array_equal(
                    self.valid_steps[written_positions],
                    valid[base_written],
                )
            ):
                raise ValueError(
                    "ground truth, context, or validity changed across archive writes"
                )
        unwritten_base = ~base_written
        if unwritten_base.any():
            new_positions = positions[unwritten_base]
            self.ground_truth[new_positions] = ground_truth_uint8[unwritten_base]
            self.context_last[new_positions] = context_last_uint8[unwritten_base]
            self.valid_steps[new_positions] = valid[unwritten_base]
            self._base_written[new_positions] = True

        actions_written = self._actions_written[mode_index, positions]
        if actions_written.any():
            written_positions = positions[actions_written]
            existing_model = self.conditioning_actions_model[
                mode_index, written_positions
            ]
            existing_cs2 = self.conditioning_actions_cs2[
                mode_index, written_positions
            ]
            if not np.array_equal(
                existing_model, model_actions_float32[actions_written]
            ) or not np.array_equal(
                existing_cs2, cs2_actions_float32[actions_written]
            ):
                raise ValueError("conditioning actions changed across evaluation seeds")
        unwritten_actions = ~actions_written
        if unwritten_actions.any():
            new_positions = positions[unwritten_actions]
            self.conditioning_actions_model[mode_index, new_positions] = (
                model_actions_float32[unwritten_actions]
            )
            self.conditioning_actions_cs2[mode_index, new_positions] = (
                cs2_actions_float32[unwritten_actions]
            )
            self._actions_written[mode_index, new_positions] = True

    def finalize(self, *, contract: dict) -> Path:
        if not self._base_written.all():
            raise ValueError("rollout archive is missing ground-truth samples")
        if not self._actions_written.all():
            raise ValueError("rollout archive is missing conditioned action streams")
        if not self._predictions_written.all():
            raise ValueError("rollout archive is missing predictions")

        arrays = (
            self.predictions,
            self.ground_truth,
            self.context_last,
            self.conditioning_actions_model,
            self.conditioning_actions_cs2,
            self.valid_steps,
        )
        for array in arrays:
            array.flush()
        del arrays
        del self.predictions
        del self.ground_truth
        del self.context_last
        del self.conditioning_actions_model
        del self.conditioning_actions_cs2
        del self.valid_steps

        artifacts = {}
        for name, final_path in self._final_paths.items():
            self._tmp_paths[name].replace(final_path)
            array = np.load(final_path, mmap_mode="r")
            artifacts[name] = {
                "path": final_path.name,
                "sha256": sha256_file(final_path),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "axes": ARRAY_SPECS[name],
            }
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "contract": {
                **contract,
                "num_samples": self.num_samples,
                "eval_seeds": self.eval_seeds,
                "action_modes": self.action_modes,
                "rollout_steps": self.rollout_steps,
                "height": self.height,
                "width": self.width,
                "num_model_actions": self.num_model_actions,
                "num_cs2_actions": self.num_cs2_actions,
                "pixel_range": "uint8_[0,255]",
            },
            "artifacts": artifacts,
        }
        metadata_path = self.out_dir / "metadata.json"
        tmp_metadata = self.out_dir / ".metadata.tmp.json"
        tmp_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        tmp_metadata.replace(metadata_path)
        return metadata_path
