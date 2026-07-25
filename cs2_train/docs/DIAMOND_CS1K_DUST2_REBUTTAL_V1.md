# DIAMOND on CounterStrike-1K 360p Dust2: rebuttal v1

This protocol supersedes the earlier pilot comparison for the rebuttal. It is
frozen before confirmatory training in
`configs/diamond_cs1k_dust2_360p_rebuttal_v1.json`.

## Question and endpoint

The experiment asks whether the DIAMOND-CSGO world model uses aligned player
actions, rather than merely predicting locally plausible video. Two models are
trained from the same initialization for exactly 50,000 optimizer steps:

- `true`: each video window receives its aligned action sequence.
- `shuffled`: each video window receives a complete action sequence from a
  different round in the minibatch. There are no fixed points. The control
  preserves sequence order and action marginals.

Both arms use one RTX PRO 6000 Blackwell 96 GB GPU, batch 80, BF16, seed 28,
the same data loader, architecture, optimizer, schedule, validation windows,
and checkpoint cadence. The test endpoint is the final step 50,000 checkpoint;
test results are not used for checkpoint selection.

## Dataset contract

The input is the materialized CounterStrike-1K 360p tier and the locked
Dust2-only manifest:

`manifest_dust2_confirmatory_spatial_v1.parquet`

SHA-256:

`33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e`

The match-disjoint split contains:

| Split | Matches | Rounds | POV rows | Aligned POV-hours |
|---|---:|---:|---:|---:|
| train | 36 | 766 | 7,660 | 87.08993 |
| val | 3 | 54 | 540 | 4.95946 |
| test | 3 | 69 | 690 | 8.29583 |

The loader consumes the already materialized MP4 and packed action files. It
uses the same optimized torchcodec decode path and action contract as the
`cs2_clean`/MIRA CounterStrike-1K loader:

- source video/actions: 32 fps;
- emitted model stream: 8 fps;
- source-frame stride: 4;
- buttons: OR over each four-frame interval;
- mouse: sum `delta_pitch` and `delta_yaw` over the interval;
- resize after decode with antialiased bilinear interpolation.

The DIAMOND adapter then maps the 12 buttons and two angular deltas to the
upstream 51-dimensional CSGO encoding.

To reproduce that direct layout from public WebDataset shards, use the checked
materializer. It filters before extraction, seeks members by the public sample
index, verifies every member hash, and records a provenance JSON:

```bash
uv run python -m cs2_train.scripts.materialize_dust2_subset \
  --source-root /data/cs1k-index \
  --shard-root /data/cs1k-shards \
  --output-root /data/cs1k-dust2-360p \
  --manifest-name manifest_dust2_confirmatory_spatial_v1.parquet \
  --sample-index-name sample_index_360p.parquet \
  --resolution 360p \
  --map-slug dust2 \
  --splits train val test
```

## Model and training contract

The copied denoiser code is pinned to upstream
`eloialonso/diamond@851cefb497733d27f1b85c804104638765860fca`
on the `csgo` branch. The full 330.5M-parameter low-resolution denoiser is
preserved:

- 4 conditioning frames;
- 4 autoregressive training steps;
- channels `[128, 256, 512, 1024]`;
- conditioning width 2048;
- depths `[2, 2, 2, 2]`;
- attention depths `[0, 0, 1, 1]`;
- model resolution 36 x 64 from the 360p source tier.

The fixed training budget is 50,000 steps x 80 sequences = 4,000,000 sampled
windows per arm. AdamW uses learning rate `1e-4`, weight decay `1e-2`, epsilon
`1e-8`, 100 warmup steps, gradient norm cap 10, and EMA decay 0.999.
Deterministic PyTorch algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are
enabled.

## Paired evaluation

Inline validation runs every 2,500 steps. True and shuffled action inputs use
the same checkpoint, fixed validation windows, and reset diffusion RNG. Both
grids and eight-step rollouts are written locally and to a private,
AES-256-encrypted S3 prefix. The owner-only review site polls a pre-signed
index; checkpoints and optimizer state are never published.

Because the GPU host uses a rotating instance-role session, a separate
`publish_signed_review_index.py` process runs under non-session reviewer
credentials. It polls the raw private index, replaces artifact links with
seven-day pre-signed URLs, and updates a stable private viewer index. The
publisher refuses temporary credentials by default so a nominal seven-day URL
cannot silently expire with a shorter role session.

After both training endpoints are complete, each final checkpoint is evaluated
on all 690 test POV rows at:

- one fixed midpoint window per POV;
- one round-shared window centered on the round's first death.

For each window, action modes `true`, `shuffled`, and `zeros` are evaluated at
seeds 37, 41, and 43. The held-out shuffle maps every target to the same POV
slot in a different round. Diffusion random draws are identical across action
modes. Per-sample JSONL is retained, and 95% percentile intervals use 10,000
bootstrap replicates clustered by `round_id`.

The primary sensitivity contrast is:

`MSE(shuffled action) - MSE(true action)`

A positive value means that correct actions improve prediction. The zero-action
contrast is reported as a secondary diagnostic.

## Reproduction

Install and test:

```bash
uv sync --extra train --extra eval
uv run --with pytest pytest -q cs2_train/tests
```

Run both arms and both confirmatory evaluations:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
DATA_DIR=/data/cs1k-360p \
RUN_ROOT=/runs/diamond-cs1k-dust2-360p-rebuttal-v1 \
bash cs2_train/scripts/run_diamond_cs1k_dust2_rebuttal_v1.sh
```

The launcher refuses a dirty tracked worktree, verifies the two dataset hashes,
and records the code commit, config, package environment, CUDA/PyTorch
environment, full `nvidia-smi -q`, exact commands, logs, metrics, checkpoints,
sample plans, and evaluator outputs.
