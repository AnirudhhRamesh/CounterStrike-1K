# CounterStrike-1K Training Baselines

This directory contains the DIAMOND-style action-conditioned video baseline
that produces the paper Table 11 / Figure 6 numbers. It consumes the same
public release artifacts as the loader and eval suite:

- `manifest.parquet`
- `round_index.parquet`
- `sample_index_360p.parquet` or `sample_index_720p.parquet`
- WebDataset shards (or the unsharded reviewer sample)

The training loop is `cs2_train/src/train.py`, runnable as a Python module.
The dataset reads parquet manifests + sample-index byte ranges directly; no
private S3 / DDB tooling is required.

## Install

From the repo root:

```bash
uv sync --extra train --extra notebooks
```

`--extra train` brings in `torch`, `torchcodec`, `torchvision`, `wandb`,
`av`, `imageio`, `imageio-ffmpeg`, and `matplotlib`. `--extra notebooks` adds
JupyterLab if you want to open `notebooks/`.

> **torchcodec note**: `torchcodec` needs a system-level FFmpeg shared library
> (`libavutil.so.*`) to import. If you see `Could not load libtorchcodec ...`,
> either install FFmpeg via your OS package manager (e.g. `apt install ffmpeg`)
> or skip the `train` extra and use eval/notebook installs — the dataset loader
> falls back to PyAV automatically when torchcodec is unavailable.

## Quick smoke run

The smallest end-to-end check (a few minutes on one GPU):

```bash
uv run python -m cs2_train.src.train \
  --config cs2_train/configs/diamond_csgo_lowres_paper.json \
  --data-dir /data/CounterStrike-1K \
  --out-dir runs/diamond_smoke \
  --action-mode true \
  --max-train-clips 8 \
  --max-val-clips 4 \
  --max-steps 100 \
  --batch-size 4 \
  --val-every 50 \
  --rollout-every 0
```

For the action-conditioning ablation, replace `--action-mode true` with
`--action-mode shuffled` or `--action-mode zeros`.

## Paper Table 11 reproduction

The completed true-action job in the paper ran for 30,000 steps on four L40S
GPUs with gradient accumulation 2 (global effective batch 128):

```bash
torchrun --nproc_per_node=4 -m cs2_train.src.train \
  --config cs2_train/configs/diamond_csgo_lowres_paper.json \
  --data-dir /data/CounterStrike-1K \
  --out-dir runs/diamond_true_4gpu_c20000 \
  --action-mode true \
  --max-train-clips 20000 --max-val-clips 512 \
  --batch-size 16 --grad-acc 2 \
  --max-steps 30000 \
  --val-every 1000 --rollout-every 5000 --ckpt-every 500
```

The shuffled-action control used the same global effective batch with
`--action-mode shuffled` over two 4-L40S nodes:

```bash
torchrun --nnodes=2 --nproc_per_node=4 -m cs2_train.src.train \
  --config cs2_train/configs/diamond_csgo_lowres_paper.json \
  --data-dir /data/CounterStrike-1K \
  --out-dir runs/diamond_shuffled_8gpu_c20000 \
  --action-mode shuffled \
  --max-train-clips 20000 --max-val-clips 512 \
  --batch-size 16 --grad-acc 1 \
  --max-steps 30000 \
  --val-every 1000 --rollout-every 5000 --ckpt-every 500
```

Both runs are reported at the matched step-20,000 evaluation boundary so the
action-mode comparison is matched by optimizer step.

## What's in this directory

```
configs/diamond_csgo_lowres_paper.json   paper-locked DIAMOND-CSGO denoiser config
src/                                      train.py / dataset.py / action_encoder.py / diamond/ / visualize.py
notebooks/                                inference + dataset + VAE exploration notebooks (optional)
DATALOADER_NOTES.md                       deeper notes on the parquet+offset dataloader
baselines/                                upstream DIAMOND baselines kept verbatim for reference
```

`scripts/run_diamond_paper_baseline.sh` is a wrapped invocation of the
`torchrun ... -m cs2_train.src.train` command above; the README intentionally
shows the explicit invocation so reviewers can see what every flag does.

## W&B logging

Set `WANDB_API_KEY` in your shell (never on the command line) and pass W&B
run flags. Logs include train loss, validation MSE/PSNR, and image/video
rollout previews:

```bash
export WANDB_API_KEY=...
uv run python -m cs2_train.src.train ... \
  --wandb-project counterstrike-1k-baselines \
  --wandb-run-name diamond-true-paper
```

Without `WANDB_API_KEY`, training falls back to plain stdout logging.
