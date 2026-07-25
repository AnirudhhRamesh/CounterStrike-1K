# CS2-WM dataloader profiling + optimization notes

Hardware under test: EC2 g6e.2xlarge (L40S 48 GB, 8 vCPU AMD EPYC) running
DLAMI Ubuntu 24, torchcodec 0.10, torch 2.10+cu128.

Data: `/opt/dlami/nvme/cs2-data` — 340 train clips (572k frames @ 30 FPS),
videos are 360×640 H.264 mp4, actions in parquet.

## Throughput

| num_workers | bs | T  | ms/batch | samples/s |
|-------------|----|----|----------|-----------|
| 0           | 16 | 6  | 2321     | 6.9       |
| 4           | 16 | 6  | 674      | 23.7      |
| 8           | 16 | 6  | 440      | 36.3      |
| 12          | 16 | 6  | 493      | 32.5      |
| 0           | 16 | 20 | 2815     | 5.7       |
| 4           | 16 | 20 | 813      | 19.7      |
| 8           | 16 | 20 | 481      | 33.3      |
| 12          | 16 | 20 | 484      | 33.1      |

Sweet spot: `num_workers=8` (= CPU count). More workers does not help —
ffmpeg decode is the bottleneck, and forkserver workers contend on the same
8 cores.

T (window length) barely affects throughput (36 vs 33 samples/s). The
torchcodec decoder caches the open file, so most of the per-call cost is in
the seek + decode-keyframe-back-to-target rather than per-frame work.

GPU throughput on the same hardware (full preset, 330M params, bs=16, bf16
autocast) is ~250 ms/microstep ≈ 64 samples/s, so the trainer is currently
**data-bound** with about 30-40% GPU idle time.

## Implemented optimizations

### Target-frame action alignment (correctness fix)
CounterStrike-1K stores each mouse delta on the frame it produces:
`action[i] == state[i] - state[i-1]`. This is the same convention used by
`cs2_clean`, whose policy windows end video history at frame `t` and supervise
the action at `t+1`. For DIAMOND at 8 fps, the transition from source video
frame `t` to `t+4` therefore aggregates action rows `t+1:t+5`: buttons are
ORed and mouse deltas are summed. The rebuttal launcher verifies both the
per-frame and four-frame identities before training and records the result in
`provenance/action_alignment_audit.json`.

The release includes a short rendered camera tail after player death. Camera
motion there can differ from the dead player's action stream, so sliding
training and midpoint validation clamp each clip at `alive_end_frame`.
First-death review videos retain the tail, while quantitative evaluator targets
after `alive_end_frame` are masked.

### Off-by-one safety in window count (correctness fix)
Some clips had `frames` in the manifest one higher than what torchcodec
actually decodes (likely encoder dropped a tail frame due to PTS rounding).
Hitting `frames - 1` raised `Invalid frame index` from torchcodec inside
DataLoader workers. We now drop the last frame as a safety margin
(`usable = n_frames - 1`), giving a deterministic, robust window count at
cost of ≤1 frame per clip. See `dataset.py: __init__`.

### Per-worker decoder + dense actions cache
`CSDataset._decoder_cache` keeps a single `VideoDecoder` open per video,
per worker, for the lifetime of the worker (which `persistent_workers=True`
makes a full epoch). `_actions_cache` keeps the densified per-frame action
array in RAM (~22 MB total for 9 hrs of training data).

### `seek_mode="approximate"` and `num_ffmpeg_threads=1`
`approximate` skips the full container scan torchcodec does in `exact` mode
on first open (1-2 s saved per worker on first touch). `num_ffmpeg_threads=1`
prevents ffmpeg from spawning extra threads that would contend with sibling
workers — we get more parallelism by running 8 workers at 1 thread each
than by running fewer workers with multi-threaded ffmpeg.

## Tried, did not work

### `decoder.get_frames_in_range(start, stop, step)`
**Theoretically** ~5-10× faster than `get_frames_at` for our consecutive
windows (sequential decode, no per-frame seeks). Locally
(single-process) it works, **but** inside DataLoader workers (Python 3.14
forkserver) it raises:

```
RuntimeError: Provided stream index=0 was not previously added.
```

`get_frames_at` works in workers, so we kept that. The bug appears to be in
the interaction between torchcodec's decoder C++ state and forkserver's
process-launch sequence. Worth retrying after a torchcodec upgrade.

## Ideas for further speedup (not implemented)

1. **Torchcodec GPU decode** (L40S has NVDEC). Would require a CUDA-enabled
   torchcodec build (`.from_url` + `device="cuda"`). With NVDEC plus
   on-device resize this could remove the CPU bottleneck entirely. Risk: GPU
   decode has overhead per-clip-open, and our access pattern hits many
   different clips per batch — net win unclear.
2. **Move resize 360×640 → 36×64 to GPU**. Currently each worker does
   `F.interpolate(...)` in CPU, which is ~50% of the per-getitem cost.
   Would cost us ~10× more host→device PCIe traffic per batch, but bs=16,
   T=6 only sums to 67 MB (≈ 4 ms over PCIe 4.0 x16).
3. **Pre-decode + re-encode** clips at 36×64 once, store on disk. Removes
   the decode-360p bottleneck entirely; trainer reads tiny mp4s.
   Trade-off: ≈ 8× less storage, but loses the ability to retrain at higher
   resolution from the same dataset.

For an overnight overnight on full preset (bs=16, 8 workers, ~36 sps), we
expect ~30-40k steps in 8 hours. That's already enough for the EMA
(decay=0.999) to converge and for the val rollout to stabilise — speeding
up the loader further is not on the critical path tonight.
