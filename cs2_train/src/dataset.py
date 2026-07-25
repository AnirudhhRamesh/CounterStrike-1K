"""CS2 video + action dataset.

Returns sliding windows of (video, actions) for any clip in the manifest.
Two output modes:

  - mode="dict" (default): returns a plain dict suitable for ad-hoc use.
  - mode="diamond": returns a `diamond.Batch` produced from the same window,
    with DIAMOND's 51-dim action encoding. Use this with `train.py`.

The manifest is the one written by `cs2_train/src/download.py`, which adds
`video_path`, `actions_path`, `match_id` aliases and a match-level `split`.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from counterstrike1k.schema import (
    ACTIONS_DTYPE as ACTIONS_BIN_DTYPE,
)
from counterstrike1k.schema import (
    BUTTONS as ACTIONS_BIN_BUTTONS,
)
from torch.utils.data import Dataset

from .action_encoder import (
    CS2_BUTTON_COLS as BUTTON_COLS,
)
from .action_encoder import (
    CS2_MOUSE_COLS as MOUSE_COLS,
)
from .action_encoder import (
    encode_cs2_actions,
)
from .diamond import Batch, Segment, SegmentId

ACTION_COLS = BUTTON_COLS + MOUSE_COLS
CS2_SOURCE_FPS = 32
# CounterStrike-1K actions are target-frame aligned: action row i contains the
# mouse delta that produced state/video frame i (state[i] - state[i - 1]).
# cs2_clean therefore pairs history ending at frame t with the action at t + 1.
# DIAMOND instead expects action t to describe the transition obs[t] -> obs[t+1],
# so every downsampled action interval begins one 32-fps source frame after its
# emitted observation.
CS2_ACTION_TARGET_OFFSET = 1
WindowMode = Literal["sliding", "midpoint", "first-death"]


class _FrameBatch:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


class PyAVVideoDecoder:
    """Small torchcodec-compatible fallback used when FFmpeg libs are missing."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def get_frames_at(self, indices: list[int]) -> _FrameBatch:
        import av

        wanted = {int(i) for i in indices}
        by_idx: dict[int, torch.Tensor] = {}
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx not in wanted:
                    continue
                arr = frame.to_ndarray(format="rgb24")
                by_idx[frame_idx] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                if len(by_idx) == len(wanted):
                    break
        missing = [idx for idx in indices if idx not in by_idx]
        if missing:
            raise IndexError(
                f"{self.path}: could not decode frame indices {missing[:5]}"
            )
        return _FrameBatch(
            torch.stack([by_idx[int(idx)] for idx in indices]).to(torch.uint8)
        )


def _local_manifest_from_metadata(data_path: Path) -> list[dict]:
    """Build a loader manifest from the locked postprocess-v2 local layout."""
    metadata_root = data_path / "metadata"
    if not metadata_root.exists():
        raise FileNotFoundError(
            f"No manifest and no metadata/ tree found under {data_path}"
        )

    manifest: list[dict] = []
    for meta_path in sorted(metadata_root.rglob("*.json")):
        with open(meta_path, "r", encoding="utf-8") as f:
            row = json.load(f)

        rel = meta_path.relative_to(metadata_root)
        rel_parent = rel.parent
        stem = meta_path.stem
        video_path = Path("videos") / rel_parent / f"{stem}.mp4"
        actions_path = Path("actions") / rel_parent / f"{stem}.actions.bin"
        if (
            not (data_path / video_path).exists()
            or not (data_path / actions_path).exists()
        ):
            continue

        row = dict(row)
        row["video_path"] = video_path.as_posix()
        row["actions_path"] = actions_path.as_posix()
        row["split"] = row.get("split") or "train"
        row.setdefault("match_id", row.get("source_demo_id", ""))
        row.setdefault("source_demo_id", row.get("match_id", ""))
        row.setdefault("player_id", row.get("pov_idx", 0))
        row.setdefault("clip_id", row.get("round_idx", 0))
        manifest.append(row)

    if not manifest:
        raise ValueError(f"No loadable metadata/action/video triples under {data_path}")
    return manifest


def _load_manifest(data_path: Path, manifest_name: str) -> list[dict]:
    manifest_path = data_path / manifest_name
    if not manifest_path.exists():
        if manifest_name == "manifest.json":
            return _local_manifest_from_metadata(data_path)
        raise FileNotFoundError(manifest_path)

    if manifest_path.suffix == ".parquet":
        return pd.read_parquet(manifest_path).to_dict("records")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_direct_files(
    root: Path, *, sample_key: str, resolution: str
) -> dict[str, bytes]:
    """Read direct-layout sidecars from a CounterStrike-1K snapshot."""

    candidates = {
        "mp4": [
            root / "videos" / resolution / f"{sample_key}.mp4",
            root / "videos" / f"{sample_key}.mp4",
        ],
        "actions.bin": [
            root / "actions" / f"{sample_key}.actions.bin",
            root / "actions" / "v12" / f"{sample_key}.actions.bin",
        ],
        "state.bin": [
            root / "state" / f"{sample_key}.state.bin",
            root / "state" / "v12" / f"{sample_key}.state.bin",
        ],
        "events.json": [
            root / "events" / f"{sample_key}.events.json",
            root / "events" / "v12" / f"{sample_key}.events.json",
        ],
        "json": [
            root / "metadata" / f"{sample_key}.json",
            root / "metadata" / "v12" / f"{sample_key}.json",
        ],
    }
    out: dict[str, bytes] = {}
    for suffix, paths in candidates.items():
        for path in paths:
            if path.exists():
                out[suffix] = path.read_bytes()
                break
        if suffix not in out:
            raise FileNotFoundError(
                f"missing direct sample {suffix} for {sample_key} under {root}"
            )
    return out


def _read_via_sample_index(
    index_path: Path,
    *,
    sample_key: str,
    resolution: str,
    root: Path,
    shard_root: Path,
    verify_sha256: bool,
) -> dict[str, bytes]:
    """Read a sample by seeking tar shard offsets recorded in `sample_index*.parquet`."""

    import hashlib

    rows = pd.read_parquet(index_path)
    rows = rows[rows["sample_key"].astype(str) == str(sample_key)]
    if "resolution" in rows.columns:
        rows = rows[rows["resolution"].astype(str) == str(resolution)]
    suffixes = {"mp4", "actions.bin", "state.bin", "events.json", "json"}
    by_suffix = {
        str(r["member_suffix"]): r
        for _, r in rows.iterrows()
        if str(r["member_suffix"]) in suffixes
    }
    missing = suffixes - set(by_suffix)
    if missing:
        raise ValueError(
            f"sample {sample_key!r} at {resolution!r} missing members {sorted(missing)}"
        )

    out: dict[str, bytes] = {}
    for suffix, row in by_suffix.items():
        shard_rel = str(row["shard_path"])
        shard_path = Path(shard_rel)
        for candidate in (
            shard_path if shard_path.is_absolute() else None,
            shard_root / shard_rel,
            root / shard_rel,
            shard_root / shard_path.name,
            root / shard_path.name,
        ):
            if candidate is None:
                continue
            if Path(candidate).exists():
                shard_path = Path(candidate)
                break
        offset = int(row["member_offset"])
        length = int(row["member_length"])
        with shard_path.open("rb") as fh:
            fh.seek(offset)
            payload = fh.read(length)
        if len(payload) != length:
            raise OSError(
                f"{shard_path}: expected {length} bytes at offset {offset}, got {len(payload)}"
            )
        expected_sha = str(row.get("member_sha256") or "")
        if verify_sha256 and expected_sha:
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected_sha:
                raise ValueError(
                    f"sha256 mismatch for {row['member_name']}: {actual} != {expected_sha}"
                )
        out[suffix] = payload
    return out


def _actions_bin_dense(actions_path: Path, clip: dict, n_frames: int) -> np.ndarray:
    arr = np.fromfile(actions_path, dtype=ACTIONS_BIN_DTYPE)
    dense = np.zeros((n_frames, len(ACTION_COLS)), dtype=np.float32)
    if len(arr) == 0 or n_frames == 0:
        return dense

    first_frame_idx = int((clip.get("actions_bin") or {}).get("first_frame_idx", 0))
    dst0 = max(0, first_frame_idx)
    src0 = max(0, -first_frame_idx)
    count = min(len(arr) - src0, n_frames - dst0)
    if count <= 0:
        return dense

    buttons = arr["buttons"][src0 : src0 + count].astype(np.uint16)
    for bit_idx, name in enumerate(ACTIONS_BIN_BUTTONS):
        if name in BUTTON_COLS:
            out_idx = BUTTON_COLS.index(name)
            dense[dst0 : dst0 + count, out_idx] = ((buttons >> bit_idx) & 1).astype(
                np.float32
            )

    mouse_offset = len(BUTTON_COLS)
    dense[dst0 : dst0 + count, mouse_offset + 0] = arr["delta_pitch"][
        src0 : src0 + count
    ].astype(np.float32)
    dense[dst0 : dst0 + count, mouse_offset + 1] = arr["delta_yaw"][
        src0 : src0 + count
    ].astype(np.float32)
    return dense


class CSDataset(Dataset):
    """Sliding-window dataset over rendered CS2 clips.

    Args:
        data_path: directory written by download.py (must contain manifest.json
            + videos/ + actions/).
        split: 'train' or 'val'. Filtered against the manifest's `split` field.
        T: window length in frames.
        stride: inter-frame stride within a window (1 = consecutive frames).
        resize: optional (H, W) target resolution. If None, native resolution
            is returned (typically 720x1280 — the open-source release default;
            older datasets may be 360x640).
        manifest_name: manifest file relative to data_path. Use this for
            balanced scaling manifests under data_path/manifests/.
        mode: 'dict' or 'diamond'. 'diamond' returns video as uint8 in [0,255]
            with DIAMOND's 51-dim action encoding wrapped in a Segment-like
            tuple for the trainer.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        T: int = 9,
        stride: int | None = None,
        target_fps: int = 8,
        resize: tuple[int, int] | None = None,
        manifest_name: str = "manifest.json",
        mode: str = "dict",
        window_mode: WindowMode = "sliding",
        map_slug: str | None = "dust2",
        resolution: str = "360p",
        shard_root: str | Path | None = None,
        subset: str | None = None,
        cache_dir: str | Path | None = None,
        max_clips: int | None = None,
        seed: int = 0,
        verify_sha256: bool = False,
    ):
        assert mode in ("dict", "diamond"), mode
        if (
            target_fps <= 0
            or target_fps > CS2_SOURCE_FPS
            or CS2_SOURCE_FPS % target_fps
        ):
            raise ValueError(
                f"CounterStrike-1K target_fps must divide {CS2_SOURCE_FPS}, got {target_fps}"
            )
        expected_stride = CS2_SOURCE_FPS // target_fps
        if stride is not None and stride != expected_stride:
            raise ValueError(
                f"stride={stride} conflicts with the {CS2_SOURCE_FPS}->{target_fps} fps "
                f"action/video contract (expected {expected_stride})"
            )
        if window_mode not in {"sliding", "midpoint", "first-death"}:
            raise ValueError(f"unknown window_mode={window_mode!r}")
        self.data_path = Path(data_path)
        self.manifest_path = self.data_path / manifest_name
        self.T = T
        self.target_fps = target_fps
        self.stride = expected_stride
        self.resize = resize
        self.mode = mode
        self.window_mode = window_mode
        self.map_slug = map_slug
        self.resolution = resolution
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else self.data_path / ".cache" / "cs2_train"
        )
        self.verify_sha256 = verify_sha256
        self.subset = subset
        self.shard_root = Path(shard_root) if shard_root is not None else self.data_path
        self.use_release_layout = False

        release_manifest = self.data_path / "manifest.parquet"
        selected_manifest = (
            release_manifest
            if manifest_name == "manifest.json" and release_manifest.exists()
            else self.manifest_path
        )
        use_release_manifest = (
            selected_manifest.suffix == ".parquet" and selected_manifest.exists()
        )
        if use_release_manifest:
            manifest_df = pd.read_parquet(selected_manifest)
            use_release_manifest = "sample_key" in manifest_df.columns
        if use_release_manifest:
            self.use_release_layout = True
            if subset:
                subset_path = self.data_path / "subsets" / f"{subset}.parquet"
                if subset_path.exists():
                    keep = set(pd.read_parquet(subset_path)["sample_key"].astype(str))
                    manifest_df = manifest_df[
                        manifest_df["sample_key"].astype(str).isin(keep)
                    ]
            if split:
                manifest_df = manifest_df[
                    manifest_df["split"].astype(str) == str(split)
                ]
            if map_slug is not None:
                if "map_slug" not in manifest_df.columns:
                    raise ValueError(f"{selected_manifest} has no map_slug column")
                manifest_df = manifest_df[
                    manifest_df["map_slug"].astype(str) == str(map_slug)
                ]
            if "fps" in manifest_df.columns:
                fps_values = set(manifest_df["fps"].astype(float).tolist())
                if fps_values != {float(CS2_SOURCE_FPS)}:
                    raise ValueError(
                        f"Expected CounterStrike-1K video/actions at {CS2_SOURCE_FPS} fps, "
                        f"found {sorted(fps_values)}"
                    )
            manifest_df = manifest_df.sort_values(
                ["split", "map_slug", "match_id", "round_idx", "pov_idx"],
                kind="stable",
            ).reset_index(drop=True)
            if max_clips is not None and len(manifest_df) > max_clips:
                rng = np.random.default_rng(seed)
                keep = np.sort(
                    rng.choice(
                        manifest_df.index.to_numpy(), size=max_clips, replace=False
                    )
                )
                manifest_df = manifest_df.loc[keep].reset_index(drop=True)
            self.samples = [
                self._release_row_to_clip(row) for _, row in manifest_df.iterrows()
            ]
        else:
            manifest = _load_manifest(self.data_path, manifest_name)
            self.samples = [
                m
                for m in manifest
                if (m.get("split") or "train") == split
                and (map_slug is None or str(m.get("map_slug")) == str(map_slug))
            ]
        if not self.samples:
            raise ValueError(f"No clips with split={split!r} in {self.manifest_path}")

        # Pre-compute window count per clip + global prefix sum.
        # We trim by 1 extra frame as a safety margin: a few of our renders
        # report `frames` in metadata one higher than what torchcodec actually
        # decodes (likely the encoder dropped a tail frame due to PTS rounding).
        # Hitting `frames - 1` raises "Invalid frame index" inside the decoder,
        # so we keep one frame's slack.
        for s in self.samples:
            n_frames = int(s.get("frames", 0))
            if n_frames <= 0:
                tick_rate = int(s.get("tick_rate", 64))
                fps = int(s.get("fps", 30))
                ticks = int(s["end_tick"]) - int(s["start_tick"])
                n_frames = round(ticks / tick_rate * fps)
            usable = max(0, n_frames - 1)  # safety: drop possibly-bad final frame
            s["num_frames"] = usable

        self._sample_by_key = {
            str(sample.get("sample_key", "")): sample for sample in self.samples
        }
        if window_mode in {"midpoint", "first-death"}:
            by_round: dict[str, list[dict]] = {}
            for sample in self.samples:
                round_key = str(
                    sample.get("round_id") or sample.get("sample_key") or id(sample)
                )
                by_round.setdefault(round_key, []).append(sample)
            for rows in by_round.values():
                common_frames = min(int(row["num_frames"]) for row in rows)
                anchor = min(
                    rows,
                    key=lambda row: int(row.get("pov_idx", row.get("player_id", 0))),
                )
                for row in rows:
                    row["_round_common_frames"] = common_frames
                    row["_round_anchor_sample_key"] = str(anchor.get("sample_key", ""))

        for s in self.samples:
            # Each emitted action is an interval summary.  Requiring T complete
            # source intervals gives every sampled frame one aligned action and
            # avoids silently truncating the final interval.
            window_frames = int(s.get("_round_common_frames", s["num_frames"]))
            max_start = (
                window_frames
                - self.T * self.stride
                - CS2_ACTION_TARGET_OFFSET
            )
            s["windows"] = (
                max(0, max_start + 1)
                if window_mode == "sliding"
                else int(max_start >= 0)
            )

        self.total_windows = sum(s["windows"] for s in self.samples)
        self.prefixes = np.cumsum([s["windows"] for s in self.samples], dtype=np.int64)

        # Per-worker caches; safe across `persistent_workers=True`.
        self.max_decoder_cache = int(
            os.environ.get("CS2_TRAIN_DECODER_CACHE_SIZE", "64")
        )
        self._decoder_cache: OrderedDict[str, object] = OrderedDict()
        self._actions_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _release_row_to_clip(row: pd.Series) -> dict:
        sample_key = str(row["sample_key"])
        return {
            **row.to_dict(),
            "_release_sample": True,
            "sample_key": sample_key,
            "source_demo_id": str(row.get("match_id", "")),
            "match_id": str(row.get("match_id", "")),
            "clip_id": int(row.get("round_idx", 0)),
            "player_id": int(row.get("pov_idx", 0)),
            "frames": int(row.get("frames", 0)),
        }

    def __len__(self) -> int:
        return int(self.total_windows)

    # ---- internal helpers ----------------------------------------------------

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)

    def _atomic_write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    def _materialize_release_sample(self, clip: dict) -> dict[str, Path]:
        direct = self._direct_release_files(str(clip["sample_key"]))
        if direct is not None:
            return direct
        sample_key = str(clip["sample_key"])
        out = self.cache_dir / self.resolution / sample_key
        files = {
            "video": out / f"{sample_key}.mp4",
            "actions": out / f"{sample_key}.actions.bin",
            "state": out / f"{sample_key}.state.bin",
            "events": out / f"{sample_key}.events.json",
            "metadata": out / f"{sample_key}.json",
        }
        marker = out / ".complete"
        if marker.exists() and all(path.exists() for path in files.values()):
            return files
        if not self.use_release_layout:
            raise RuntimeError(
                "release sample requested but no CounterStrike-1K release manifest found at data_path"
            )
        payload = self._read_release_sample_payload(sample_key)
        self._atomic_write_bytes(files["video"], payload["mp4"])
        self._atomic_write_bytes(files["actions"], payload["actions.bin"])
        self._atomic_write_bytes(files["state"], payload["state.bin"])
        self._atomic_write_text(files["events"], payload["events.json"].decode("utf-8"))
        self._atomic_write_text(files["metadata"], payload["json"].decode("utf-8"))
        self._atomic_write_text(marker, "ok\n")
        return files

    def _direct_release_files(self, sample_key: str) -> dict[str, Path] | None:
        """Resolve an already materialized release row without copying payloads."""

        candidates = {
            "video": [
                self.data_path / "videos" / self.resolution / f"{sample_key}.mp4",
                self.data_path / "videos" / f"{sample_key}.mp4",
            ],
            "actions": [
                self.data_path / "actions" / f"{sample_key}.actions.bin",
                self.data_path / "actions" / "v12" / f"{sample_key}.actions.bin",
            ],
            "state": [
                self.data_path / "state" / f"{sample_key}.state.bin",
                self.data_path / "state" / "v12" / f"{sample_key}.state.bin",
            ],
            "events": [
                self.data_path / "events" / f"{sample_key}.events.json",
                self.data_path / "events" / "v12" / f"{sample_key}.events.json",
            ],
            "metadata": [
                self.data_path / "metadata" / f"{sample_key}.json",
                self.data_path / "metadata" / "v12" / f"{sample_key}.json",
            ],
        }
        resolved: dict[str, Path] = {}
        for name, paths in candidates.items():
            match = next((path for path in paths if path.is_file()), None)
            if match is None:
                return None
            resolved[name] = match
        return resolved

    def _read_release_sample_payload(self, sample_key: str) -> dict[str, bytes]:
        """Read one sample's WebDataset members from a local snapshot.

        Supports both the unsharded direct-file layout (preview repo) and the
        sample_index parquet + tar offsets layout (full WDS repo).
        """

        index_path = (
            self.data_path / f"sample_index_{self.resolution}.parquet"
            if (self.data_path / f"sample_index_{self.resolution}.parquet").exists()
            else self.data_path / "sample_index.parquet"
        )
        if index_path.exists():
            return _read_via_sample_index(
                index_path,
                sample_key=sample_key,
                resolution=self.resolution,
                root=self.data_path,
                shard_root=self.shard_root,
                verify_sha256=self.verify_sha256,
            )
        return _read_direct_files(
            self.data_path, sample_key=sample_key, resolution=self.resolution
        )

    def _get_decoder(self, video_path: Path):
        key = str(video_path)
        decoder = self._decoder_cache.get(key)
        if decoder is None:
            try:
                from torchcodec.decoders import VideoDecoder

                # seek_mode="approximate" avoids the full container scan torchcodec
                # does in "exact" mode — safe here because locked renders are CFR 32 FPS.
                decoder = VideoDecoder(
                    key, seek_mode="approximate", num_ffmpeg_threads=1
                )
            except Exception as exc:  # noqa: BLE001
                reason = (
                    str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                )
                warnings.warn(
                    "torchcodec VideoDecoder is unavailable; falling back to PyAV. "
                    "Install matching FFmpeg shared libraries for high-throughput DIAMOND training. "
                    f"Original error: {exc.__class__.__name__}: {reason}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                decoder = PyAVVideoDecoder(key)
            self._decoder_cache[key] = decoder
            if self.max_decoder_cache > 0:
                while len(self._decoder_cache) > self.max_decoder_cache:
                    _, old_decoder = self._decoder_cache.popitem(last=False)
                    close = getattr(old_decoder, "close", None)
                    if callable(close):
                        close()
            elif self.max_decoder_cache == 0:
                self._decoder_cache.pop(key, None)
        else:
            self._decoder_cache.move_to_end(key)
        return decoder

    def _get_actions_dense(self, clip: dict) -> np.ndarray:
        """Densify the parquet to a full (num_frames, 14) float32 array.

        New CS2-WM shards expose the locked 12-button + 2-mouse schema. The
        fallback path keeps older pilot renders readable by zero-filling any
        missing action column; the paper baseline logs this in CHANGES.md and
        should run only on locked-schema data.
        """
        if clip.get("_release_sample"):
            actions_path = self._materialize_release_sample(clip)["actions"]
            key = str(actions_path)
        else:
            key = clip["actions_path"]
            actions_path = self.data_path / key
        cached = self._actions_cache.get(key)
        if cached is not None:
            return cached
        n = clip["num_frames"]
        if actions_path.name.endswith(".actions.bin"):
            dense = _actions_bin_dense(actions_path, clip, n)
        else:
            try:
                df = pd.read_parquet(actions_path, columns=["frame_idx", *ACTION_COLS])
            except Exception:  # noqa: BLE001
                df = pd.read_parquet(actions_path)
                missing = [c for c in ACTION_COLS if c not in df.columns]
                if missing:
                    warnings.warn(
                        f"{actions_path} is missing action columns {missing}; zero-filling for compatibility",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    for c in missing:
                        df[c] = 0.0
                df = df[["frame_idx", *ACTION_COLS]]
            dense = np.zeros((n, len(ACTION_COLS)), dtype=np.float32)
            fidx = df["frame_idx"].to_numpy()
            vals = df[ACTION_COLS].to_numpy(dtype=np.float32)
            mask = (fidx >= 0) & (fidx < n)
            dense[fidx[mask]] = vals[mask]
        self._actions_cache[key] = dense
        return dense

    def _resolve_window(self, global_idx: int) -> tuple[dict, int, list[int]]:
        if global_idx < 0 or global_idx >= self.total_windows:
            raise IndexError(global_idx)
        clip_idx = int(np.searchsorted(self.prefixes, global_idx, side="right"))
        clip = self.samples[clip_idx]
        prev = 0 if clip_idx == 0 else int(self.prefixes[clip_idx - 1])
        if self.window_mode == "sliding":
            local_start = int(global_idx - prev)
        else:
            max_start = (
                int(clip.get("_round_common_frames", clip["num_frames"]))
                - self.T * self.stride
                - CS2_ACTION_TARGET_OFFSET
            )
            if self.window_mode == "midpoint":
                local_start = max_start // 2
            else:
                local_start = self._first_death_start(clip, max_start)
        frame_ids = list(
            range(local_start, local_start + self.T * self.stride, self.stride)
        )
        return clip, local_start, frame_ids

    def _first_death_start(self, clip: dict, max_start: int) -> int:
        if max_start < 0:
            raise ValueError(
                f"{clip['sample_key']} is too short for the requested window"
            )
        anchor_key = str(
            clip.get("_round_anchor_sample_key", clip.get("sample_key", ""))
        )
        anchor_clip = self._sample_by_key.get(anchor_key, clip)
        if anchor_clip.get("_release_sample"):
            events_path = self._materialize_release_sample(anchor_clip)["events"]
        else:
            events_path = self.data_path / anchor_clip["events_path"]
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        events = payload.get("events", []) if isinstance(payload, dict) else payload
        anchors = sorted(
            int(event["frame_idx"])
            for event in events
            if event.get("type") == "player_death"
            and "frame_idx" in event
            and 0
            <= int(event["frame_idx"])
            < int(clip.get("_round_common_frames", clip["num_frames"]))
        )
        if not anchors:
            raise ValueError(f"{clip['sample_key']} has no in-range player_death event")
        span = self.T * self.stride
        return min(max(anchors[0] - span // 2, 0), max_start)

    def _aggregate_actions(self, dense: np.ndarray, start: int) -> np.ndarray:
        """Aggregate target-aligned actions for emitted 32->target-fps transitions.

        If emitted observation ``i`` is source frame ``start + i * stride``,
        its action interval is the following source frames
        ``[obs_frame + 1, obs_frame + stride]`` (inclusive). This matches
        cs2_clean's history-at-t/action-at-t+1 convention and ensures the mouse
        deltas integrate exactly from the emitted observation to the next one.
        """

        action_start = start + CS2_ACTION_TARGET_OFFSET
        action_stop = action_start + self.T * self.stride
        raw = dense[action_start:action_stop]
        if raw.shape[0] != self.T * self.stride:
            raise ValueError(
                f"action window [{action_start}:{action_stop}] has {raw.shape[0]} records, "
                f"expected {self.T * self.stride}"
            )
        windows = raw.reshape(self.T, self.stride, len(ACTION_COLS))
        buttons = windows[:, :, : len(BUTTON_COLS)].max(axis=1)
        mouse = windows[:, :, len(BUTTON_COLS) :].sum(axis=1)
        return np.concatenate((buttons, mouse), axis=1).astype(np.float32, copy=False)

    def action_window_at(self, global_idx: int) -> tuple[torch.Tensor, dict]:
        """Load only the encoded action window and provenance for an index.

        Confirmatory action-shuffle evaluation uses this path for donor actions
        so it does not decode donor videos a second time.
        """

        clip, local_start, frame_ids = self._resolve_window(global_idx)
        dense = self._get_actions_dense(clip)
        aggregated = self._aggregate_actions(dense, local_start)
        encoded = encode_cs2_actions(
            aggregated[:, : len(BUTTON_COLS)],
            aggregated[:, len(BUTTON_COLS) :],
        )
        return torch.from_numpy(encoded), {
            "sample_key": str(clip.get("sample_key", "")),
            "match_id": str(clip["match_id"]),
            "round_id": str(clip.get("round_id", "")),
            "pov_idx": int(clip.get("pov_idx", clip.get("player_id", 0))),
            "source_start_frame": local_start,
            "source_frame_indices": frame_ids,
            "source_action_frame_intervals": [
                [frame_idx + CS2_ACTION_TARGET_OFFSET, frame_idx + self.stride]
                for frame_idx in frame_ids
            ],
            "source_fps": CS2_SOURCE_FPS,
            "target_fps": self.target_fps,
            "window_mode": self.window_mode,
            "dataset_index": int(global_idx),
        }

    # ---- entry points --------------------------------------------------------

    def __getitem__(self, global_idx: int):
        clip, local_start, frame_ids = self._resolve_window(global_idx)

        # Video: torchcodec returns uint8 [T,C,H,W]
        if clip.get("_release_sample"):
            video_path = self._materialize_release_sample(clip)["video"]
        else:
            video_path = self.data_path / clip["video_path"]
        decoder = self._get_decoder(video_path)
        # NOTE: get_frames_in_range would be faster for consecutive windows, but
        # in our DataLoader (forkserver multiprocessing on Python 3.14) it raises
        # "Provided stream index=0 was not previously added" inside workers. We
        # stick with get_frames_at, which works under workers.
        fb = decoder.get_frames_at(indices=frame_ids)
        frames_u8 = fb.data  # uint8

        if self.resize is not None:
            # Bilinear interpolate in float, then cast back to uint8 for diamond mode.
            f = frames_u8.float()
            f = F.interpolate(
                f,
                size=self.resize,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            frames_u8 = f.round().clamp_(0, 255).to(torch.uint8)

        # Actions: one interval-aligned record per emitted 8-fps video frame.
        dense = self._get_actions_dense(clip)
        aggregated = self._aggregate_actions(dense, local_start)
        act10 = torch.from_numpy(aggregated.copy())

        if self.mode == "dict":
            video = frames_u8.float().div_(127.5).sub_(1.0)
            return {
                "video": video,  # [T,C,H,W] in [-1,1]
                "actions": act10,  # [T,14]
                "clip_idx": int(
                    np.searchsorted(self.prefixes, global_idx, side="right")
                ),
                "start_frame": local_start,
                "meta": {
                    "match_id": clip["match_id"],
                    "pov_idx": int(clip.get("pov_idx", clip.get("player_id", 0))),
                    "clip_id": clip["clip_id"],
                },
            }

        # diamond mode: DIAMOND expects obs in [-1,1] float and act as multi-hot
        obs = frames_u8.float().div_(127.5).sub_(1.0)  # [T,3,H,W]
        act, action_info = self.action_window_at(global_idx)
        T = obs.size(0)
        rew = torch.zeros(T, dtype=torch.float32)
        end = torch.zeros(T, dtype=torch.uint8)
        trunc = torch.zeros(T, dtype=torch.uint8)
        mask_padding = torch.ones(T, dtype=torch.bool)
        seg_id = SegmentId(
            episode_id=f"{clip['source_demo_id']}/pov_{int(clip.get('pov_idx', clip.get('player_id', 0))):02d}/{clip['clip_id']}",
            start=local_start,
            stop=local_start + T * self.stride,
        )
        return Segment(
            obs=obs,
            act=act,
            rew=rew,
            end=end,
            trunc=trunc,
            mask_padding=mask_padding,
            info=action_info,
            id=seg_id,
        )


# ---------- collate -----------------------------------------------------------


def collate_diamond(segments: list[Segment]) -> Batch:
    """Stack a list of Segment into a DIAMOND Batch."""
    return Batch(
        obs=torch.stack([s.obs for s in segments]),
        act=torch.stack([s.act for s in segments]),
        rew=torch.stack([s.rew for s in segments]),
        end=torch.stack([s.end for s in segments]),
        trunc=torch.stack([s.trunc for s in segments]),
        mask_padding=torch.stack([s.mask_padding for s in segments]),
        info=[s.info for s in segments],
        segment_ids=[s.id for s in segments],
    )
