# Changes from upstream DIAMOND-CSGO

This document is the required W5 change log for the CS2-WM DIAMOND baseline.

## Upstream source

- Repository: `https://github.com/eloialonso/diamond`
- Branch: `csgo`
- Baseline config copied into: `configs/diamond_csgo_lowres_paper.json`

## Architecture changes

None for the B1 low-resolution denoiser. The copied `src/diamond/*` modules
preserve the upstream denoiser, inner model, UNet blocks, and EDM sampler.

## Training hyperparameter changes

None in the paper config. `configs/diamond_csgo_lowres_paper.json` pins the
upstream CSGO denoiser training values:

- 4 conditioning frames
- 4 autoregressive training steps
- batch size 16
- gradient accumulation 8
- accumulated losses are not divided by gradient accumulation before backward
- learning rate `1e-4`
- weight decay `1e-2`
- warmup 100 optimizer steps
- max grad norm 10
- 240,000 optimizer steps for a full run
- no EMA
- no mixed precision

Shorter runs or bf16 runs are allowed only as pilot/debug runs and should not
be reported as the B1 paper result.

## Data/input adapter changes

The data loader is replaced because CS2-WM stores rendered MP4 video and action
parquet files instead of DIAMOND's processed HDF5 episodes.

The adapter returns the same DIAMOND `Batch` fields:

- `obs`: frames in `[-1, 1]`
- `act`: 51-dim DIAMOND CSGO multi-hot action vector
- `rew`, `end`, `trunc`: zeros, because B1 is a pure world-model baseline
- `mask_padding`: valid-frame mask

Resolution is `36x64` to preserve CS2-WM's 16:9 aspect ratio.

## Action mapping changes

DIAMOND's 51-dim CSGO action space is preserved. CS2-WM's locked 12-button
schema is mapped into that space:

| CS2-WM field | DIAMOND slot |
|---|---|
| `FORWARD` | `w` |
| `LEFT` | `a` |
| `BACK` | `s` |
| `RIGHT` | `d` |
| `JUMP` | `space` |
| `DUCK` | `ctrl` |
| `WALK` | `shift` |
| `RELOAD` | `r` |
| `FIRE` | left click |
| `RIGHTCLICK` | right click |
| `delta_yaw` | DIAMOND mouse-x bucket |
| `delta_pitch` | DIAMOND mouse-y bucket |

`INSPECT`, `USE`, `active_weapon`, and `health` are not consumed by B1 because
they have no upstream DIAMOND-CSGO action-slot equivalent. This is an input
adapter limitation, not a model change.

## Evaluation changes

Upstream DIAMOND-CSGO evaluation is primarily interactive world-model playback
plus component test loss. CS2-WM adds deterministic dataset-paper evaluation:

- held-out denoising loss
- one-step MSE/PSNR
- autoregressive rollout MSE/PSNR per step
- generated rollout videos for FVD

FVD computation is intentionally separate because it depends on an external
video embedding model. Do not report FVD until the script has run on the fixed
held-out split and written the value to `metrics.json`.
