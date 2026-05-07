# CounterStrike-1K Training Baselines

This directory contains the DIAMOND-style action-conditioned video baseline
used for release smoke tests and paper-facing pilots. It consumes the same
public release artifacts as the loader and eval suite:

- `manifest.parquet`
- `round_index.parquet`
- `sample_index_360p.parquet` or `sample_index_720p.parquet`
- local WebDataset shards from a Hugging Face snapshot, or direct sample files

The loader is `cs2_train/src/dataset.py`. When `--data-dir` contains
`manifest.parquet`, it uses the public `counterstrike1k.CounterStrike1K` API,
materializes selected samples into `--cache-dir`, decodes video windows, and
maps v12 CS2 controls into DIAMOND's 51-dimensional CSGO action space.

## Release Smoke

```bash
uv run python cs2_train/src/train.py \
  --data-dir /data/CounterStrike-1K \
  --shard-root /data/CounterStrike-1K-360-wds \
  --resolution 360p \
  --cache-dir /data/cs2-diamond-cache \
  --preset small \
  --max-train-clips 8 \
  --max-val-clips 4 \
  --max-steps 100 \
  --batch-size 4 \
  --num-workers 2 \
  --val-every 50 \
  --rollout-every 0 \
  --wandb-project counterstrike-1k-evals \
  --wandb-run-name diamond-release-smoke
```

For the action-conditioning ablation, run the same command with
`--action-mode shuffled` or `--action-mode zeros`. The main baseline uses
`--action-mode true`.

## Throughput Notes

`torchcodec` is preferred for high-throughput training. If matching FFmpeg
shared libraries are unavailable, the loader falls back to PyAV so small smoke
runs still work, but full training should install compatible FFmpeg/torchcodec
or use a pre-materialized local shard/sample cache.

Maintainer-only S3 range reads are supported for fresh private shard smoke tests
with `COUNTERSTRIKE1K_S3_BUCKET=counterstrike-1k`. Public reviewers should use
local Hugging Face snapshots instead.
