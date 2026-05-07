"""Video decoding helpers for short CounterStrike-1K windows."""

from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
from PIL import Image
from PIL import ImageDraw


def _fit_label(draw: ImageDraw.ImageDraw, label: str, max_width: int) -> str:
    if not label or draw.textlength(label) <= max_width:
        return label
    suffix = "..."
    available = max(0, max_width - int(draw.textlength(suffix)))
    trimmed = label
    while trimmed and draw.textlength(trimmed) > available:
        trimmed = trimmed[:-1]
    return trimmed.rstrip() + suffix


def _frame_index_from_pts(frame, stream, fps: float, fallback_idx: int) -> int:
    if frame.pts is None or stream.time_base is None:
        return fallback_idx
    seconds = float(frame.pts * stream.time_base)
    return int(round(seconds * fps))


def decode_sampled_frames(
    video_bytes: bytes,
    frame_indices: list[int] | np.ndarray,
    *,
    resize: tuple[int, int] | None = (224, 224),
) -> np.ndarray:
    """Decode selected frame indices from an MP4 payload.

    Returns uint8 frames shaped ``(T, H, W, 3)``. The function first tries a
    timestamp-indexed decode after seeking near the requested window. If MP4
    timestamps are not aligned as expected, it falls back to a full sequential
    decode using decoded-frame order.
    """

    wanted = sorted({int(i) for i in frame_indices if int(i) >= 0})
    if not wanted:
        raise ValueError("frame_indices must contain at least one non-negative index")
    frames = _decode_by_pts(video_bytes, wanted, resize=resize)
    if len(frames) == len(wanted):
        return np.stack([frames[i] for i in wanted], axis=0)

    frames = _decode_by_order(video_bytes, wanted, resize=resize)
    missing = [i for i in wanted if i not in frames]
    if missing:
        raise ValueError(f"could not decode requested frames {missing[:8]} from MP4")
    return np.stack([frames[i] for i in wanted], axis=0)


def decode_sampled_frames_from_path(
    video_path: str | Path,
    frame_indices: list[int] | np.ndarray,
    *,
    resize: tuple[int, int] | None = (224, 224),
) -> np.ndarray:
    return decode_sampled_frames(Path(video_path).read_bytes(), frame_indices, resize=resize)


def sample_frame_indices(start_frame: int, end_frame: int, num_frames: int) -> list[int]:
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    n = max(1, min(int(num_frames), int(end_frame - start_frame)))
    if n == 1:
        return [int(start_frame)]
    values = np.linspace(int(start_frame), int(end_frame) - 1, n)
    return sorted({int(round(v)) for v in values})


def make_frame_grid(
    frames: list[np.ndarray],
    *,
    labels: list[str] | None = None,
    columns: int = 5,
    pad: int = 4,
    label_height: int = 16,
) -> np.ndarray:
    if not frames:
        raise ValueError("frames must be non-empty")
    columns = max(1, int(columns))
    labels = labels or ["" for _ in frames]
    images = [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in frames]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * pad,
            rows * (height + label_height) + (rows + 1) * pad,
        ),
        color=(16, 16, 16),
    )
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = pad + col * (width + pad)
        y = pad + row * (height + label_height + pad)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        canvas.paste(image, (x, y + label_height))
        if labels[idx]:
            label = _fit_label(draw, str(labels[idx]), max(1, width - 4))
            draw.text((x + 2, y), label, fill=(240, 240, 240))
    return np.asarray(canvas, dtype=np.uint8)


def _to_rgb_array(frame, resize: tuple[int, int] | None) -> np.ndarray:
    img = frame.to_image().convert("RGB")
    if resize is not None:
        img = img.resize(resize, Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _decode_by_pts(
    video_bytes: bytes,
    wanted: list[int],
    *,
    resize: tuple[int, int] | None,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate is not None else 32.0
        start = min(wanted)
        max_wanted = max(wanted)
        if start > 0 and stream.time_base is not None:
            timestamp = int((start / fps) / stream.time_base)
            container.seek(max(0, timestamp), stream=stream, backward=True)
        fallback_idx = 0
        wanted_set = set(wanted)
        for frame in container.decode(stream):
            idx = _frame_index_from_pts(frame, stream, fps, fallback_idx)
            fallback_idx += 1
            if idx in wanted_set:
                out[idx] = _to_rgb_array(frame, resize)
            if idx > max_wanted + max(8, int(round(fps))):
                break
    return out


def _decode_by_order(
    video_bytes: bytes,
    wanted: list[int],
    *,
    resize: tuple[int, int] | None,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    wanted_set = set(wanted)
    max_wanted = max(wanted)
    with av.open(io.BytesIO(video_bytes)) as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in wanted_set:
                out[idx] = _to_rgb_array(frame, resize)
            if idx > max_wanted:
                break
    return out
