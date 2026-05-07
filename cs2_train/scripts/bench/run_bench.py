"""Data-loader micro-benchmark for CounterStrike-1K.

Compares two loaders end-to-end on the same clips, on the same hardware:

  - raw_mp4:    CSDataset (existing production path) — per-clip mp4 + parquet
                from local disk, torchcodec decoder cache per worker.
  - webdataset: streaming tar shards (also local disk for parity), bytes
                decoded inside the worker via torchcodec from BytesIO + pyarrow.

Each loader is exercised in four modes:

  io_breakdown      median per-stage time (open|seek+decode|materialise|actions),
                    cold vs warm, fixed T=8.
  throughput        full DataLoader samples/s, sweep num_workers ∈ {0,1,2,4,8,12}
                    with bs=16, T=8.
  T_sweep           num_workers=8 fixed, sweep T ∈ {1,4,8,16,30}, bs=16.
  window_latency    num_workers=0 (single-process), sweep T to expose the per-frame
                    decode slope vs the per-clip open cost.

Writes one JSON per mode under <out_dir>. Plotting lives in plot_bench.py.

Usage:
    uv run python -m cs2_train.scripts.bench.run_bench \
        --raw-root  /opt/dlami/nvme/bench/raw \
        --shard-glob '/opt/dlami/nvme/bench/shards/shard-*.tar' \
        --out-dir   /opt/dlami/nvme/bench/results \
        --modes     io_breakdown throughput T_sweep window_latency

Run takes ~10 min on g6e.2xlarge.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import statistics
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchcodec.decoders import VideoDecoder

# Make the project's cs2_train package importable when run as a script.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_train.src.dataset import CSDataset  # noqa: E402

ACTION_COLS = [
    "FORWARD", "BACK", "LEFT", "RIGHT", "JUMP", "DUCK", "WALK",
    "FIRE", "RIGHTCLICK", "RELOAD", "INSPECT", "USE",
    "delta_pitch", "delta_yaw",
]

# ---------- WebDataset loader ----------------------------------------------------

class WdsClipDataset(IterableDataset):
    """Stream tar shards; for each sample, decode an mp4 window + parquet actions.

    Yields the same dict shape as CSDataset(mode='dict') so we can collate either
    one with the same downstream code:
        {video: [T,3,H,W] in [-1,1], actions: [T,14], meta: {...}}

    Because we want to compare loader throughput end-to-end, the per-sample work
    here matches CSDataset: pick a random T-frame window inside the clip's
    decodable range, decode via torchcodec, parse the parquet, densify to
    per-frame actions, slice to the window.
    """

    def __init__(
        self,
        shard_paths: list[str],
        T: int = 8,
        stride: int = 1,
        shuffle_buffer: int = 100,
        seed: int = 0,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.T = T
        self.stride = stride
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _iter_samples_from_shard(self, shard_path: str):
        """Yield raw {key: bytes} samples from one tar."""
        # Iterate sequentially; the OS does the prefetch.
        with tarfile.open(shard_path, "r|") as tar:
            current_basename = None
            current_parts: dict[str, bytes] = {}
            for member in tar:
                if not member.isfile():
                    continue
                # webdataset key convention: split on first dot
                name = member.name
                base, _, key = name.partition(".")
                if current_basename is not None and base != current_basename and current_parts:
                    yield current_parts
                    current_parts = {}
                current_basename = base
                f = tar.extractfile(member)
                if f is None:
                    continue
                current_parts[key] = f.read()
            if current_parts:
                yield current_parts

    def _decode_sample(self, parts: dict[str, bytes], rng: random.Random):
        """Decode a window from a sample's mp4 bytes + parse its parquet."""
        video_bytes = parts["mp4"]
        # torchcodec accepts bytes via from-bytes constructor in 0.10
        decoder = VideoDecoder(
            video_bytes,
            seek_mode="approximate",
            num_ffmpeg_threads=1,
        )
        n = decoder.metadata.num_frames
        usable = max(1, n - 1)
        window = self.T * self.stride
        if usable < window:
            return None
        local_start = rng.randint(0, usable - window)
        frame_ids = list(range(local_start, local_start + window, self.stride))
        fb = decoder.get_frames_at(indices=frame_ids)
        frames_u8 = fb.data  # [T,3,H,W] uint8

        # actions
        act_bytes = parts.get("actions.parquet")
        if act_bytes is None:
            act = torch.zeros((self.T, len(ACTION_COLS)), dtype=torch.float32)
        else:
            df = pq.read_table(io.BytesIO(act_bytes), columns=["frame_idx", *ACTION_COLS]).to_pandas()
            dense = np.zeros((n, len(ACTION_COLS)), dtype=np.float32)
            fidx = df["frame_idx"].to_numpy()
            vals = df[ACTION_COLS].to_numpy(dtype=np.float32)
            mask = (fidx >= 0) & (fidx < n)
            dense[fidx[mask]] = vals[mask]
            act = torch.from_numpy(dense[frame_ids].copy())

        video = frames_u8.float().div_(127.5).sub_(1.0)
        return {"video": video, "actions": act}

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + wid + int(time.time()) // 60)
        # Shard-by-worker: each worker takes a disjoint subset of shards.
        if worker is not None:
            shards = [s for i, s in enumerate(self.shard_paths) if i % worker.num_workers == wid]
        else:
            shards = self.shard_paths
        if not shards:
            return
        # tiny shuffle buffer for sample-level mix-up
        buf: list[dict] = []
        order = list(range(len(shards)))
        rng.shuffle(order)
        for s_idx in order:
            for parts in self._iter_samples_from_shard(shards[s_idx]):
                buf.append(parts)
                if len(buf) >= self.shuffle_buffer:
                    j = rng.randrange(len(buf))
                    parts = buf.pop(j)
                    out = self._decode_sample(parts, rng)
                    if out is not None:
                        yield out
        rng.shuffle(buf)
        for parts in buf:
            out = self._decode_sample(parts, rng)
            if out is not None:
                yield out


def _collate(batch):
    return {
        "video":   torch.stack([b["video"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
    }


# ---------- Timing helpers -------------------------------------------------------

def _ns():
    return time.perf_counter_ns()

def _ms(ns_diff):
    return ns_diff / 1_000_000.0

def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = sorted(values)
    return {
        "n":      len(arr),
        "mean":   round(sum(arr) / len(arr), 4),
        "median": round(arr[len(arr) // 2], 4),
        "p05":    round(arr[max(0, int(0.05 * len(arr)) - 1)], 4),
        "p95":    round(arr[min(len(arr) - 1, int(0.95 * len(arr)))], 4),
        "min":    round(arr[0], 4),
        "max":    round(arr[-1], 4),
    }


# ---------- IO BREAKDOWN ---------------------------------------------------------

def bench_io_breakdown(args, *, T: int = 8, n_samples: int = 200, seed: int = 0):
    """Per-stage timing on raw_mp4 vs webdataset, cold and warm."""
    out: dict[str, dict] = {"T": T, "n_samples": n_samples}

    # ---------- RAW MP4 ----------
    raw_root = Path(args.raw_root)
    manifest = json.loads((raw_root / "manifest.json").read_text())
    rng = random.Random(seed)
    eligible = [c for c in manifest if int(c.get("frames", 0)) > T + 1]
    # Sample with replacement so most picks land on already-cached decoders;
    # a uniform shuffle over 150 clips would give 0 warm hits at n_samples=250.
    samples = rng.choices(eligible, k=n_samples)

    # Each sample: open VideoDecoder (cold), seek+decode T frames, materialize, parse parquet.
    open_ms, sd_ms, mat_ms, act_ms = [], [], [], []
    open_warm_ms, sd_warm_ms = [], []
    decoders: dict[str, VideoDecoder] = {}
    parquet_dense: dict[str, np.ndarray] = {}

    for clip in samples:
        v_path = str(raw_root / clip["video_path"])
        a_path = str(raw_root / clip["actions_path"])
        n_frames = max(1, int(clip.get("frames", 0)) - 1)
        if n_frames < T + 1:
            continue
        local_start = rng.randint(0, n_frames - T - 1)
        frame_ids = list(range(local_start, local_start + T))

        # Cold open: never seen before
        is_warm = v_path in decoders
        t0 = _ns()
        if is_warm:
            decoder = decoders[v_path]
        else:
            decoder = VideoDecoder(v_path, seek_mode="approximate", num_ffmpeg_threads=1)
            decoders[v_path] = decoder
        t1 = _ns()
        # Seek + decode
        fb = decoder.get_frames_at(indices=frame_ids)
        t2 = _ns()
        # Materialise
        video = fb.data.float().div_(127.5).sub_(1.0)
        t3 = _ns()
        # Actions: cold parse + slice
        if a_path not in parquet_dense:
            df = pq.read_table(a_path, columns=["frame_idx", *ACTION_COLS]).to_pandas()
            dense = np.zeros((n_frames + 1, len(ACTION_COLS)), dtype=np.float32)
            fidx = df["frame_idx"].to_numpy()
            vals = df[ACTION_COLS].to_numpy(dtype=np.float32)
            mask = (fidx >= 0) & (fidx < dense.shape[0])
            dense[fidx[mask]] = vals[mask]
            parquet_dense[a_path] = dense
        dense = parquet_dense[a_path]
        act = torch.from_numpy(dense[frame_ids].copy())
        t4 = _ns()

        if is_warm:
            open_warm_ms.append(_ms(t1 - t0))
            sd_warm_ms.append(_ms(t2 - t1))
        else:
            open_ms.append(_ms(t1 - t0))
            sd_ms.append(_ms(t2 - t1))
        mat_ms.append(_ms(t3 - t2))
        act_ms.append(_ms(t4 - t3))

    out["raw_mp4"] = {
        "open_cold_ms":      _summary(open_ms),
        "open_warm_ms":      _summary(open_warm_ms),
        "seek_decode_cold_ms": _summary(sd_ms),
        "seek_decode_warm_ms": _summary(sd_warm_ms),
        "materialize_ms":    _summary(mat_ms),
        "actions_ms":        _summary(act_ms),
    }
    cold_p50 = out['raw_mp4']['open_cold_ms'].get('median', 'n/a')
    warm_p50 = out['raw_mp4']['seek_decode_warm_ms'].get('median', 'n/a')
    print(f"[raw_mp4] cold open p50={cold_p50} ms, warm seek+decode p50={warm_p50} ms")

    # ---------- WEBDATASET ----------
    shards = sorted(glob.glob(args.shard_glob))
    if not shards:
        out["webdataset"] = {"error": "no shards matched"}
    else:
        wds_open_ms, wds_extract_ms, wds_decode_ms, wds_mat_ms, wds_act_ms = [], [], [], [], []
        n_done = 0
        for shard in shards:
            if n_done >= n_samples:
                break
            t_open0 = _ns()
            tar = tarfile.open(shard, "r|")
            t_open1 = _ns()
            wds_open_ms.append(_ms(t_open1 - t_open0))

            current_basename = None
            current_parts: dict[str, bytes] = {}
            for member in tar:
                if not member.isfile():
                    continue
                base, _, key = member.name.partition(".")
                if current_basename is not None and base != current_basename and current_parts:
                    # Process the completed sample
                    parts = current_parts
                    current_parts = {}
                    if "mp4" not in parts:
                        current_basename = base
                        f = tar.extractfile(member)
                        current_parts[key] = f.read() if f else b""
                        continue

                    # extract was already done as we read; just take the cost
                    # of the in-memory mp4-decode + actions parse.
                    t_dec0 = _ns()
                    decoder = VideoDecoder(parts["mp4"], seek_mode="approximate", num_ffmpeg_threads=1)
                    n = decoder.metadata.num_frames
                    if n < T + 1:
                        current_basename = base
                        f = tar.extractfile(member)
                        current_parts[key] = f.read() if f else b""
                        continue
                    local_start = rng.randint(0, max(0, n - T - 1))
                    frame_ids = list(range(local_start, local_start + T))
                    fb = decoder.get_frames_at(indices=frame_ids)
                    t_dec1 = _ns()
                    video = fb.data.float().div_(127.5).sub_(1.0)
                    t_mat = _ns()
                    if "actions.parquet" in parts:
                        df = pq.read_table(io.BytesIO(parts["actions.parquet"]),
                                           columns=["frame_idx", *ACTION_COLS]).to_pandas()
                        dense = np.zeros((n, len(ACTION_COLS)), dtype=np.float32)
                        fidx = df["frame_idx"].to_numpy()
                        vals = df[ACTION_COLS].to_numpy(dtype=np.float32)
                        mask = (fidx >= 0) & (fidx < n)
                        dense[fidx[mask]] = vals[mask]
                        act = torch.from_numpy(dense[frame_ids].copy())
                    t_act = _ns()
                    wds_decode_ms.append(_ms(t_dec1 - t_dec0))
                    wds_mat_ms.append(_ms(t_mat - t_dec1))
                    wds_act_ms.append(_ms(t_act - t_mat))
                    n_done += 1
                    if n_done >= n_samples:
                        break
                current_basename = base
                f = tar.extractfile(member)
                t_x0 = _ns()
                data = f.read() if f else b""
                t_x1 = _ns()
                wds_extract_ms.append(_ms(t_x1 - t_x0))
                current_parts[key] = data
            tar.close()

        out["webdataset"] = {
            "shard_open_ms":    _summary(wds_open_ms),
            "tar_extract_ms":   _summary(wds_extract_ms),
            "decode_ms":        _summary(wds_decode_ms),
            "materialize_ms":   _summary(wds_mat_ms),
            "actions_ms":       _summary(wds_act_ms),
        }
        print(f"[webdataset] decode p50={out['webdataset']['decode_ms']['median']} ms, "
              f"tar_extract p50={out['webdataset']['tar_extract_ms']['median']} ms")

    return out


# ---------- THROUGHPUT (full DataLoader) -----------------------------------------

def _make_raw_loader(raw_root: Path, T: int, batch_size: int, num_workers: int):
    ds = CSDataset(data_path=raw_root, split="train", T=T, stride=1, mode="dict")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate,
    )

def _make_wds_loader(shard_glob: str, T: int, batch_size: int, num_workers: int):
    shards = sorted(glob.glob(shard_glob))
    ds = WdsClipDataset(shards, T=T, stride=1, shuffle_buffer=100)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate,
    )


def _measure_loader(loader, *, target_seconds: float = 20.0, max_batches: int | None = None):
    """Return (samples_per_s, batches, seconds)."""
    n_samples = 0
    n_batches = 0
    t0 = _ns()
    for batch in loader:
        # Touch the tensor so any deferred work runs.
        batch["video"].size(0)
        bs = batch["video"].size(0)
        n_samples += bs
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
        if _ms(_ns() - t0) > target_seconds * 1000:
            break
    elapsed_ns = _ns() - t0
    secs = elapsed_ns / 1e9
    return {
        "samples": n_samples,
        "batches": n_batches,
        "seconds": round(secs, 3),
        "samples_per_s": round(n_samples / max(secs, 1e-6), 3),
        "ms_per_batch": round(secs * 1000.0 / max(n_batches, 1), 3),
    }


def bench_throughput(args, *, T: int = 8, batch_size: int = 16,
                     workers_grid=(0, 1, 2, 4, 8, 12),
                     target_seconds: float = 20.0):
    out = {"T": T, "batch_size": batch_size, "target_seconds": target_seconds, "results": {}}

    for nw in workers_grid:
        out["results"].setdefault("raw_mp4", {})
        out["results"].setdefault("webdataset", {})
        # warm-up (one batch) — pays decoder open cost
        for fmt, mk in (("raw_mp4", _make_raw_loader), ("webdataset", _make_wds_loader)):
            print(f"[throughput] fmt={fmt} workers={nw}")
            try:
                if fmt == "raw_mp4":
                    loader = mk(Path(args.raw_root), T, batch_size, nw)
                else:
                    loader = mk(args.shard_glob, T, batch_size, nw)
                # warm-up
                _measure_loader(loader, target_seconds=2.0, max_batches=3)
                # measure
                r = _measure_loader(loader, target_seconds=target_seconds)
                out["results"][fmt][str(nw)] = r
                print(f"  {fmt:>10}/w={nw}: {r['samples_per_s']:.2f} sps, "
                      f"{r['ms_per_batch']:.1f} ms/batch")
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR fmt={fmt} workers={nw}: {e!r}")
                out["results"][fmt][str(nw)] = {"error": repr(e)}
            finally:
                # ensure workers shut down
                del loader

    return out


# ---------- T_SWEEP --------------------------------------------------------------

def bench_T_sweep(args, *, batch_size: int = 16, num_workers: int = 8,
                  Ts=(1, 4, 8, 16, 30), target_seconds: float = 20.0):
    out = {"batch_size": batch_size, "num_workers": num_workers, "results": {}}
    for T in Ts:
        out["results"].setdefault("raw_mp4", {})
        out["results"].setdefault("webdataset", {})
        for fmt, mk in (("raw_mp4", _make_raw_loader), ("webdataset", _make_wds_loader)):
            print(f"[T_sweep] fmt={fmt} T={T}")
            try:
                if fmt == "raw_mp4":
                    loader = mk(Path(args.raw_root), T, batch_size, num_workers)
                else:
                    loader = mk(args.shard_glob, T, batch_size, num_workers)
                _measure_loader(loader, target_seconds=2.0, max_batches=3)
                r = _measure_loader(loader, target_seconds=target_seconds)
                r["frames_per_s"] = round(r["samples_per_s"] * T, 2)
                out["results"][fmt][str(T)] = r
                print(f"  {fmt:>10} T={T}: {r['samples_per_s']:.2f} sps "
                      f"({r['frames_per_s']:.0f} fps)")
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR fmt={fmt} T={T}: {e!r}")
                out["results"][fmt][str(T)] = {"error": repr(e)}
            finally:
                del loader
    return out


# ---------- WINDOW LATENCY (single-process, isolate decoder) ---------------------

def bench_window_latency(args, *, n_samples: int = 200, Ts=(1, 4, 8, 16, 30), seed: int = 0):
    out = {"n_samples": n_samples, "results": {}}

    raw_root = Path(args.raw_root)
    manifest = json.loads((raw_root / "manifest.json").read_text())
    rng = random.Random(seed)

    # Re-use a single decoder cache across T-sweeps to isolate per-frame work.
    decoders: dict[str, VideoDecoder] = {}

    for T in Ts:
        eligible = [c for c in manifest if int(c.get("frames", 0)) > T + 1]
        rng.shuffle(eligible)
        # Warm-up: open the first 30 decoders to remove cold-open noise from the latency curve.
        for clip in eligible[:30]:
            v_path = str(raw_root / clip["video_path"])
            if v_path not in decoders:
                decoders[v_path] = VideoDecoder(v_path, seek_mode="approximate", num_ffmpeg_threads=1)
        timings = []
        for clip in eligible[: n_samples + 30]:
            v_path = str(raw_root / clip["video_path"])
            n_frames = max(1, int(clip["frames"]) - 1)
            if n_frames < T + 1:
                continue
            local_start = rng.randint(0, n_frames - T - 1)
            frame_ids = list(range(local_start, local_start + T))
            if v_path not in decoders:
                decoders[v_path] = VideoDecoder(v_path, seek_mode="approximate", num_ffmpeg_threads=1)
            dec = decoders[v_path]
            t0 = _ns()
            fb = dec.get_frames_at(indices=frame_ids)
            _ = fb.data.shape  # touch
            t1 = _ns()
            timings.append(_ms(t1 - t0))
            if len(timings) >= n_samples:
                break
        out["results"][str(T)] = _summary(timings)
        print(f"[latency] T={T}: p50={out['results'][str(T)]['median']} ms, "
              f"p95={out['results'][str(T)]['p95']} ms")

    return out


# ---------- Main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--shard-glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", nargs="+",
                    default=["io_breakdown", "throughput", "T_sweep", "window_latency"],
                    choices=["io_breakdown", "throughput", "T_sweep", "window_latency"])
    ap.add_argument("--target-seconds", type=float, default=20.0,
                    help="Per-config measurement window (throughput/T_sweep)")
    ap.add_argument("--n-samples-io", type=int, default=200)
    ap.add_argument("--workers-grid", nargs="+", type=int,
                    default=[0, 1, 2, 4, 8, 12])
    ap.add_argument("--Ts", nargs="+", type=int, default=[1, 4, 8, 16, 30])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch DataLoader needs forkserver to play nicely with torchcodec under Py3.14.
    torch.multiprocessing.set_start_method("forkserver", force=True)
    # Reduce thread fan-out — match the production loader's discipline.
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if "io_breakdown" in args.modes:
        r = bench_io_breakdown(args, T=8, n_samples=args.n_samples_io)
        (out_dir / "io_breakdown.json").write_text(json.dumps(r, indent=2))
        print(f"wrote {out_dir / 'io_breakdown.json'}")

    if "throughput" in args.modes:
        r = bench_throughput(args, workers_grid=args.workers_grid,
                             target_seconds=args.target_seconds)
        (out_dir / "throughput.json").write_text(json.dumps(r, indent=2))
        print(f"wrote {out_dir / 'throughput.json'}")

    if "T_sweep" in args.modes:
        r = bench_T_sweep(args, Ts=args.Ts, target_seconds=args.target_seconds)
        (out_dir / "T_sweep.json").write_text(json.dumps(r, indent=2))
        print(f"wrote {out_dir / 'T_sweep.json'}")

    if "window_latency" in args.modes:
        r = bench_window_latency(args, Ts=args.Ts)
        (out_dir / "window_latency.json").write_text(json.dumps(r, indent=2))
        print(f"wrote {out_dir / 'window_latency.json'}")


if __name__ == "__main__":
    main()
