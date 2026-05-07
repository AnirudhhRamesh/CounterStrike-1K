# DIAMOND-CSGO baseline protocol for CS2-WM

This is the paper-facing B1 baseline: upstream DIAMOND-CSGO low-resolution
world model trained on CS2-WM. The goal is baseline fidelity, not tuning.

## Upstream pin

Source: `eloialonso/diamond`, branch `csgo`.

Pinned files:

- `config/agent/csgo.yaml`
- `config/trainer.yaml`
- `config/env/csgo.yaml`
- `config/world_model_env/fast.yaml`
- `src/models/diffusion/*`

The checked-in config is `configs/diamond_csgo_lowres_paper.json`.

## What must match upstream

- Low-res denoiser architecture:
  - `cond_channels=2048`
  - `channels=[128, 256, 512, 1024]`
  - `depths=[2, 2, 2, 2]`
  - `attn_depths=[0, 0, 1, 1]`
  - `num_steps_conditioning=4`
  - `sigma_data=0.5`
  - `sigma_offset_noise=0.1`
  - `noise_previous_obs=true`
- Training:
  - `num_autoregressive_steps=4`
  - `batch_size=16`
  - `grad_acc_steps=8`
  - accumulated losses are not divided by `grad_acc_steps` before backward,
    matching the upstream trainer implementation
  - `lr=1e-4`
  - `weight_decay=1e-2`
  - `eps=1e-8`
  - `lr_warmup_steps=100`
  - `max_grad_norm=10.0`
  - no EMA
  - no mixed precision in the paper config
- Training length:
  - upstream config is 600 epochs x 400 optimizer steps = 240,000 optimizer
    steps. Shorter runs must be labeled pilot/ablation, not the B1 table.
- Fast sampler for validation rollouts:
  - `num_steps_denoising=1`
  - `sigma_min=2e-3`
  - `sigma_max=5.0`
  - `rho=7`
  - `order=1`
  - `s_churn=0`
  - `s_cond=0.005`

## Allowed CounterStrike-1K adapters

- Data source: CounterStrike-1K public release roots:
  `manifest.parquet`, `sample_index_360p.parquet` / `sample_index_720p.parquet`,
  and local WebDataset shards or direct reviewer sample files. Maintainer-only
  S3 range reads are allowed for smoke tests, but paper runs should use local
  snapshots/caches. All paths must emit the same `diamond.Batch` contract.
- Resolution: CounterStrike-1K is 16:9, so the low-res input is `36x64` rather than
  DIAMOND-CSGO's `150x280` source shape or `48x64` processed shape.
- Action adapter: preserve DIAMOND's 51-dim action space. Map compatible
  CS2 controls into DIAMOND key/click slots:
  - `FORWARD -> w`
  - `LEFT -> a`
  - `BACK -> s`
  - `RIGHT -> d`
  - `JUMP -> space`
  - `DUCK -> ctrl`
  - `WALK -> shift`
  - `RELOAD -> r`
  - `FIRE -> left click`
  - `RIGHTCLICK -> right click`
  - `INSPECT`, `USE`, `active_weapon`, and `health` are not used by B1.

## Action-conditioning ablations

Use `--action-mode true` for the main action-conditioned baseline. For the
one-day NeurIPS ablation, run identical short-budget jobs with
`--action-mode shuffled` and optionally `--action-mode zeros`; both modes keep
the video batches fixed and alter only the DIAMOND action tensor. Report these
as ablations, not as separately tuned models.

## Evaluation for paper table

Run `src.evaluate` on the held-out split with the checkpoint selected by the
pre-declared criterion. For B1, use the lowest validation denoising/one-step
MSE checkpoint unless the paper explicitly reports final-step checkpoints.

Report:

- dataset version and split file hash
- checkpoint path and SHA256
- train optimizer steps and effective batch
- denoising loss on held-out windows
- one-step MSE and PSNR
- 16-frame or 32-frame autoregressive rollout MSE/PSNR per step
- rollout MSE mean and last-step MSE
- FVD over generated vs. real rollouts once `cd-fvd` is installed
- qualitative rollout grid path used in the paper figure

FVD is the primary generative video metric for the paper, but it needs an
external video embedding model. `src.evaluate` writes rollout MP4 folders that
can be passed to `cd-fvd`; it does not silently invent an FVD number when the
dependency or model weights are unavailable.

## Current EC2 pilot data caveat

The current `/opt/dlami/nvme/cs2-data` manifest is a 520-clip, 3-demo pilot
with missing map names and a random match-level train/val split. It is useful
for code validation and convergence checks, but it is not the Dust2-only,
fixed-split dataset specified for the NeurIPS v1 paper. Numbers from that
manifest must be labeled as pilot numbers.
