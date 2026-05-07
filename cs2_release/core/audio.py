"""Audio decoding helpers for CounterStrike-1K MP4 windows.

The release MP4s are H.264 video + stereo AAC at 44.1 kHz. The audio probe
treats audio as a frozen modality, so we only need a deterministic mono
waveform decoder for short windows.
"""

from __future__ import annotations

import io

import av
import numpy as np


def decode_audio_window(
    video_bytes: bytes,
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Decode one mono float32 audio window from an MP4 byte payload.

    The decoded waveform is downmixed to mono by averaging available channels,
    resampled to ``sample_rate`` via PyAV's resampler, and hard-trimmed/zero
    padded to exactly ``round(duration_seconds * sample_rate)`` samples.

    Decodes from the start of the stream rather than seeking — release MP4s are
    short clips (tens of seconds) so the cost is small, and avoiding seek means
    the resampler keeps a single coherent state across the whole stream.
    """

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if start_seconds < 0:
        raise ValueError("start_seconds must be >= 0")

    target_samples = int(round(float(duration_seconds) * int(sample_rate)))
    if target_samples <= 0:
        raise ValueError("duration_seconds * sample_rate must round to >= 1 sample")

    end_seconds = float(start_seconds) + float(duration_seconds)
    end_samples = int(round(end_seconds * int(sample_rate)))

    with av.open(io.BytesIO(video_bytes)) as container:
        if not container.streams.audio:
            raise ValueError("MP4 payload has no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=int(sample_rate))

        chunks: list[np.ndarray] = []
        total_samples = 0
        finished = False
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                if resampled is None:
                    continue
                array = resampled.to_ndarray()
                if array.ndim == 2:
                    array = array.mean(axis=0)
                array = array.astype(np.float32, copy=False)
                chunks.append(array)
                total_samples += int(array.shape[0])
            if total_samples >= end_samples + int(sample_rate * 0.05):
                finished = True
                break
        if not finished:
            for resampled in resampler.resample(None):
                if resampled is None:
                    continue
                array = resampled.to_ndarray()
                if array.ndim == 2:
                    array = array.mean(axis=0)
                chunks.append(array.astype(np.float32, copy=False))

    if not chunks:
        return np.zeros(target_samples, dtype=np.float32)
    waveform = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    start_sample = int(round(float(start_seconds) * int(sample_rate)))
    if start_sample >= len(waveform):
        return np.zeros(target_samples, dtype=np.float32)
    waveform = waveform[start_sample:start_sample + target_samples]
    if len(waveform) < target_samples:
        out = np.zeros(target_samples, dtype=np.float32)
        out[: len(waveform)] = waveform
        return out
    return waveform.astype(np.float32, copy=False)


def waveform_to_log_mel(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16_000,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    fmin: float = 0.0,
    fmax: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute a deterministic log-mel spectrogram from a 1-D waveform.

    Returns ``(n_mels, n_time)`` float32 array. Uses ``torch.stft`` with a Hann
    window plus a NumPy mel filterbank so the only runtime dependency beyond
    NumPy is a torch import. Output is log10(power + eps).
    """

    import torch  # local import keeps the helper usable in CPU-only contexts.

    if waveform.ndim != 1:
        raise ValueError("waveform must be 1-D mono")
    if waveform.size == 0:
        raise ValueError("waveform is empty")
    fmax_resolved = float(fmax) if fmax is not None else float(sample_rate) / 2.0

    window = torch.hann_window(int(n_fft))
    spec = torch.stft(
        torch.from_numpy(waveform.astype(np.float32, copy=False)),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(n_fft),
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
        normalized=False,
    )
    power = (spec.real ** 2 + spec.imag ** 2).numpy().astype(np.float32, copy=False)
    fb = _mel_filterbank(
        n_mels=int(n_mels),
        n_fft=int(n_fft),
        sample_rate=int(sample_rate),
        fmin=float(fmin),
        fmax=fmax_resolved,
    )
    mel = fb @ power
    return np.log10(np.maximum(mel, eps)).astype(np.float32, copy=False)


def _mel_filterbank(
    *,
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Slaney-style mel filterbank as a ``(n_mels, n_fft//2 + 1)`` matrix."""

    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(np.int64)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, centre, right = int(bin_pts[m - 1]), int(bin_pts[m]), int(bin_pts[m + 1])
        if centre > left:
            ramp = np.arange(left, centre, dtype=np.float32) - left
            fb[m - 1, left:centre] = ramp / max(centre - left, 1)
        if right > centre:
            ramp = right - np.arange(centre, right, dtype=np.float32)
            fb[m - 1, centre:right] = ramp / max(right - centre, 1)
    enorm = 2.0 / np.maximum(hz_pts[2 : n_mels + 2] - hz_pts[: n_mels], 1e-12)
    fb *= enorm[:, None]
    return fb
