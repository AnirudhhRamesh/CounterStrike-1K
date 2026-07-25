# DIAMOND on CounterStrike-1K 360p Dust2: rebuttal v1

This protocol supersedes the earlier pilot comparison for the rebuttal. It is
frozen before confirmatory training in
`configs/diamond_cs1k_dust2_360p_rebuttal_v1.json`.

The corrected production training/evaluation code is frozen at
`34524f6b6f1f805200d72ab4e77f3a55dd6415f8`.

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
- release actions are target-frame aligned, matching `cs2_clean` history
  through frame `t` / action at frame `t+1`;
- for an emitted observation at source frame `t`, buttons are ORed and mouse
  deltas summed over source action rows `t+1` through `t+4` (inclusive), which
  exactly describes the transition to the next observation at `t+4`;
- sliding training and midpoint validation stop at `alive_end_frame`; the
  rendered post-death camera tail is not player-controlled and is not part of
  the aligned-action training hours;
- resize after decode with antialiased bilinear interpolation.

The DIAMOND adapter then maps the 12 buttons and two angular deltas to the
upstream 51-dimensional CSGO encoding.

Before either arm starts, `audit_cs1k_action_alignment.py` checks a
deterministic train/validation/test panel. It requires action and state ticks
to match, verifies `action[i].mouse == state[i] - state[i-1]`, and verifies
that each four-row aggregate equals the corresponding 8-fps state transition.
The audit is enforced over the released alive interval and separately reports
post-alive camera/action mismatches. The machine-readable result is retained in
`provenance/action_alignment_audit.json`.

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

The first-death stress window retains post-death frames in qualitative review
videos, but quantitative rollout targets at or beyond a POV's
`alive_end_frame` are masked. One-step targets remain pre-death for all rows.
Thus camera motion after control has ended cannot dilute the action contrast.

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
runs the target-frame action-alignment audit, and records the code commit,
config, package environment, CUDA/PyTorch environment, full `nvidia-smi -q`,
exact commands, logs, metrics, checkpoints, sample plans, and evaluator
outputs.

## Final audit and private review publication

The training launcher is deliberately frozen at the preregistered training
commit. After its four confirmatory evaluations finish, run the cross-arm
audit from the current analysis commit:

```bash
python -m cs2_train.scripts.summarize_diamond_rebuttal \
  --run-root /runs/diamond-cs1k-dust2-360p-rebuttal-v1 \
  --expected-step 50000 \
  --expected-samples 690
```

The audit refuses to summarize an incomplete or mismatched experiment. It
requires the pinned training commit, the declared arm identity, matched
training hyperparameters, one final checkpoint per arm, distinct checkpoint
hashes between arms, and the same manifest, evaluator config, checkpoint step,
sample plan, held-out action-donor plan, action modes, rollout settings, and
evaluation seeds across the two checkpoint arms. It writes:

- `evaluation/rebuttal_summary.json`, with machine-readable checkpoint means,
  within-checkpoint action sensitivity, true-versus-shuffled training effects,
  the action-sensitivity difference in differences, and the complete paired
  2,500-step inline validation/rollout trajectory for both training arms;
- `evaluation/rebuttal_summary.md`, with a compact table of the same
  round-clustered 95% bootstrap intervals.

The difference in differences is the primary causal diagnostic:

`[MSE(shuffled input) - MSE(true input)]_true-trained`
`- [MSE(shuffled input) - MSE(true input)]_shuffled-trained`

A positive value means that aligned-action training increased sensitivity to
the correct held-out action sequence beyond any sensitivity learned by the
shuffled control.

The inline trajectory is diagnostic only. At each checkpoint, true and
shuffled actions share identical diffusion draws, but the deterministic sampler
seed changes with checkpoint step. This makes every within-step action contrast
paired while avoiding a claim that absolute MSE changes between inline
checkpoints are a fixed-noise learning curve. The paper-facing endpoint instead
uses the preregistered step-50,000 checkpoints, three fixed evaluation seeds,
and all held-out test rows.

For the secondary “train longer?” decision, evaluate the saved checkpoints on
validation data only:

```bash
python -m cs2_train.scripts.run_diamond_validation_checkpoint_audit \
  --run-root /runs/diamond-cs1k-dust2-360p-rebuttal-v1 \
  --data-dir /data/cs1k-360p \
  --steps 10000 20000 30000 40000 50000 \
  --pov-idx 0 \
  --expected-samples 54
```

This audit selects the same POV slot from every one of the 54 validation
rounds and uses identical midpoint windows, action donors, batch order, and
diffusion seeds at all five checkpoints for both training arms. It reports the
round-clustered change in action sensitivity from step 40,000 to 50,000. It
never reads the test split and cannot alter the primary 50,000-step endpoint.

Only after both audits succeed, publish the sanitized summaries, convergence
report, and a bounded set of review videos to the existing private S3 index:

```bash
python -m cs2_train.scripts.publish_diamond_final_review \
  --run-root /runs/diamond-cs1k-dust2-360p-rebuttal-v1 \
  --bucket cs2-wm-rollout-preview-377114445113 \
  --prefix diamond-cs1k/rebuttal-v1 \
  --run-id diamond-cs1k-dust2-360p-rebuttal-v1 \
  --step 50000 \
  --videos-per-eval 4
```

The publisher verifies that the bucket blocks public access, enables AES-256
server-side encryption for every upload, and never uploads model checkpoints,
optimizer state, or per-sample data. The long-lived signing service then
refreshes the owner-only viewer index with seven-day pre-signed artifact URLs.
