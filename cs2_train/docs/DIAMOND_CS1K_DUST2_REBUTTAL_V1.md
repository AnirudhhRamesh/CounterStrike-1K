# DIAMOND on CounterStrike-1K 360p Dust2: rebuttal v1

This protocol supersedes the earlier pilot comparison for the rebuttal. It is
frozen before confirmatory training in
`configs/diamond_cs1k_dust2_360p_rebuttal_v1.json`.

The corrected production training/evaluation code is frozen at
`34524f6b6f1f805200d72ab4e77f3a55dd6415f8`.

## Pre-test 20k resource amendment

On 2026-07-25 UTC, with approximately 12 hours remaining before the rebuttal
evidence deadline and before opening the confirmatory test split, the compute
plan was shortened. The already-running aligned-action trajectory stops at its
atomic step-20,000 checkpoint. The separately trained shuffled-action arm is
not run. This decision is based on elapsed compute and the submission
deadline, not held-out test quality.

The original 50,000-step, two-training-arm contract below is retained as the
superseded plan. The machine-readable amendment is
`configs/diamond_cs1k_dust2_360p_20k_resource_amendment_v1.json`. The amended
endpoint evaluates the single aligned-trained checkpoint on all 69 test
rounds / 690 POVs under true, cross-round shuffled, and zero action inputs at
midpoint and first-death windows with seeds 37, 41, and 43. Videos and
diffusion random draws remain paired, and both the rollout archive and fixed
RAFT motion endpoint are retained.

The amended claim is strictly within-checkpoint action-conditioning
sensitivity at step 20,000. It is not a matched trained-model comparison, a
difference-in-differences result, the original 50,000-step endpoint, or a
full-budget upstream DIAMOND reproduction. The step-20,000 training trajectory
itself is unchanged from the original frozen config.

### Completed resource-amended endpoint (2026-07-26 UTC)

The atomic step-20,000 checkpoint has SHA-256
`d921ca28468f594e3f0136b9ed89262fe449ac73a8540149ca2886ef29368080`.
Both windows completed over all 69 held-out rounds / 690 POV rows and all
three fixed evaluation seeds:

| Window | Metric | True | Cross-round shuffled | Shuffled - true | Round-cluster 95% CI | Positive rounds |
|---|---|---:|---:|---:|---:|---:|
| midpoint | eight-step mean MSE | 0.080541 | 0.122850 | +0.042309 | [0.037105, 0.047518] | 68/69 |
| midpoint | world-view RAFT EPE | 0.035250 | 0.047394 | +0.012144 | [0.011147, 0.013180] | 69/69 |
| first-death | eight-step mean MSE | 0.089599 | 0.128401 | +0.038803 | [0.034877, 0.042928] | 67/69 |
| first-death | world-view RAFT EPE | 0.036859 | 0.049624 | +0.012765 | [0.011672, 0.013871] | 69/69 |

Correct actions therefore reduce world-view flow EPE by 25.6% at midpoint and
25.7% around first death relative to the distribution-matched cross-round
control. Pixel MSE agrees but is secondary because it can reward blurry or
conservative predictions.

The frozen 4.15M-parameter temporal action probe was trained on 200,000
real-video train segments and selected at epoch 14 on 20,000 match-disjoint
validation segments (validation macro average precision `0.633439`). Its
checkpoint SHA-256 is
`c8080ef0ec8fede4fc935d07d4eac9b3840d82d8cbc0b00669b73dc42f433b00`.
Midpoint true-target alignment separation is `+0.040302`, CI
`[0.017990, 0.084380]`. The first-death estimate is `+0.011330`, CI
`[-0.071565, 0.029523]`, and is inconclusive; 146 post-alive segments are
excluded. The first-death ARR result must not be described as a replication
of the significant midpoint ARR result.

An additive all-action resummary at code commit
`b8d36e130796292c26cdb862cfae31cdea936e94` recomputes the original ARR
results exactly from their hashed score/label arrays, then reports the same
10,000-draw round bootstrap for every one of the 14 frozen action labels. It
does not rerun or select the world model, backbone, or probe. Selected
midpoint rows are shown below; the machine-readable addendum contains all
labels.

| Action label | True-target ARR | Shuffled-target ARR | Zero-target ARR | True - shuffled (95% CI) | Positive segments |
|---|---:|---:|---:|---:|---:|
| FIRE | 0.296 | 0.214 | 0.156 | +0.083 [-0.012, 0.202] | 291 |
| RIGHTCLICK | 0.248 | 0.200 | 0.200 | +0.049 [0.006, 0.130] | 189 |
| YAW_NEG | 0.710 | 0.576 | 0.620 | +0.134 [0.067, 0.193] | 1,263 |
| YAW_POS | 0.691 | 0.614 | 0.657 | +0.078 [0.016, 0.137] | 1,119 |
| RELOAD | 0.016 | 0.015 | 0.016 | +0.002 [-0.103, 0.513] | 15 |

These per-label intervals are an exhaustive post-test diagnostic, not a new
multiple-testing-corrected confirmatory endpoint. FIRE is directionally
ordered and its true-versus-zero interval is positive
(`[0.043, 0.250]`), but its true-versus-shuffled interval overlaps zero.
RELOAD has only 15 positive segments and is explicitly unsupported. The
predeclared macro separation above and RAFT motion endpoint remain the
inferential claims.

Reproduce the additive table without visual rescoring:

```bash
python -m cs2_train.scripts.resummarize_rollout_arr \
  /runs/diamond-cs1k-dust2-360p-rebuttal-v2/evaluation_20k/midpoint/action_recoverability \
  --sample-plan /runs/diamond-cs1k-dust2-360p-rebuttal-v2/evaluation_20k/midpoint/sample_plan.json \
  --output /runs/diamond-cs1k-dust2-360p-rebuttal-v2/evaluation_20k/midpoint/action_recoverability/per_action_addendum.json
```

The real midpoint addendum SHA-256 is
`d0aeb93e8cc2353b675c062acdd949d6b26c4c93ee60a64d7715f8e1d632e69e`;
the first-death addendum SHA-256 is
`66ddb1a93e84bafa4c6272020b1d5e47abc243abd3844da6f9c641fdbf9b9589`.

The ARR-extended independent audit is public at commit
`d138f57707cb15fc0ee5ecf0d5023cdb360bbd8d`. It rehashes the checkpoint,
both rollout archives, all pixel/RAFT rows, the probe and train/validation
feature archives, and both ARR outputs. The real audit passed with report
SHA-256
`874be977f8576111067c39b60328732ce477109d3c7bb87339f96449794e4724`.

```bash
python cs2_train/scripts/audit_diamond_20k_endpoint.py \
  --endpoint-root /runs/diamond-cs1k-dust2-360p-rebuttal-v2/evaluation_20k \
  --checkpoint /runs/diamond-cs1k-dust2-360p-rebuttal-v2/true/step_0020000.pt \
  --probe-checkpoint /runs/temporal-arr-cs1k-dust2-v1/probe/temporal_action_probe.pt \
  --output /runs/diamond-cs1k-dust2-360p-rebuttal-v2/evaluation_20k/post_test_integrity_audit_with_arr.json
```

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

Each arm is launched in a fresh process. The global CPU and CUDA RNGs are
reset to seed 28 before dataset and model construction, while action
derangements use a separate CPU `torch.Generator` seeded with 90001. A release
test requires a shuffled-action draw to advance only that dedicated generator,
not the global model/data RNG. Consequently, changing the arm changes the
action intervention without changing the seeded initialization or data-order
RNG stream.

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

The adapter vendors this contract rather than importing `cs2_clean`, so the
release has no private runtime dependency. Both loaders use manifest-indexed
overlapping windows, TorchCodec random-access decoding with a PyAV fallback,
per-worker decoder LRU caches, pinned-memory transfer, and persistent workers.
The frozen DIAMOND run uses eight workers and prefetch factor four; it also
caches densified actions and can seek public WebDataset members by the release
sample index before atomically materializing them. These additions implement
DIAMOND's 8-fps transition and public-shard requirements without changing the
underlying CounterStrike-1K sample or action semantics.

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

### Training-budget classification

This is a controlled matched-compute action-sensitivity study, not a
full-budget reproduction of upstream DIAMOND-CSGO. The upstream paper config
uses 240,000 optimizer steps at effective batch 128, or 30.72 million sampled
sequences. The frozen study uses 4.00 million sequences per arm, approximately
13.0% of that sequence exposure.

The confirmatory claim is limited to the paired aligned-versus-shuffled action
contrast under the fixed 50k budget. The 50k result must not be labeled as the
full upstream B1 baseline. A full-budget reproduction or a matched post-50k
extension is a separately preregistered experiment and cannot replace this
endpoint.

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

### Pre-test dynamics-metric amendment

On 2026-07-25, before any held-out test evaluation and before the first
numbered training checkpoint, qualitative validation review showed that pixel
MSE can obscure correct action-conditioned camera motion when the generated
frame is blurry or slightly misregistered. The original MSE endpoint above is
retained and cannot be replaced. The following secondary endpoints are added
before test access, following the separation used by the MIRA technical
report between visual quality, physical-state recovery, and action
recoverability:

1. **Camera/motion dynamics agreement.** A fixed torchvision RAFT-Small
   `C_T_V2` model estimates adjacent optical flow for the generated and
   ground-truth rollout. Inputs are resized from the model-native 36 x 64
   output to 128 x 224 with bilinear interpolation and no antialiasing, as in
   the torchvision inference recipe. The scorer measures dense endpoint error
   and robust median global-flow endpoint error, normalized by the resized
   image diagonal. It scores the normalized crop `y=[0.10,0.68]`,
   `x=[0.04,0.96]` to remove most static HUD and first-person weapon pixels.
   Post-alive steps use the same mask as MSE. The fixed weights SHA-256 is
   `01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27`.
2. **CS2 Action Recoverability Ratio (ARR).** A frozen visual backbone plus
   temporal probe is trained on real train windows and selected only on
   validation. It detects the action stream from generated motion and divides
   generated average precision by the corresponding model-native real-video
   ceiling. Button and signed mouse-look scores are reported separately; the
   common-action macro excludes `INSPECT` and `USE`, which the upstream
   DIAMOND 51-dimensional action space cannot represent.
3. **State/dynamics probe.** A shared real-video probe recovers camera
   yaw/pitch, player position, active weapon, and ammo from generated
   trajectories. This covers state changes that optical flow cannot, while
   keeping the evaluator common across DIAMOND, NanoWM, and MIRA-single.

RAFT measures camera and visible scene motion, not all game physics: muzzle
flashes, firing, reloads, and weapon switches require ARR/state/event metrics.
SSIM and LPIPS remain appearance diagnostics. Frame FID ignores time, while
FVD/FDD measure marginal temporal realism rather than whether the requested
action produced the correct paired transition; they may be reported as
secondary quality metrics but are not substitutes for controllability.

The expensive sampler is replayed once from the frozen final checkpoints with
`--save-rollout-archive`. It atomically retains standard NumPy arrays for every
seed x action mode x sample: last context frame, generated and ground-truth
future, alive mask, actual 51-D model input, and canonical 14-D CS2 action
stream. Every array is hashed and `metadata.json` is written last as the
completion marker. The replay MSE and sample/action-plan hashes must match the
original confirmatory evaluation before any new metric is accepted.

## Reproduction

Install and test:

```bash
uv sync --frozen --extra train --extra eval
uv run --frozen --with pytest==9.1.1 pytest -q cs2_train/tests
```

Pull requests touching the DIAMOND adapter run this exact command on Python
3.12 in `.github/workflows/diamond-cs1k-tests.yml`. The workflow pins every
third-party GitHub Action by commit SHA and pins the `uv` and `pytest`
versions, then builds the source distribution and wheel twice and requires
byte-identical archives.

Run both arms and both confirmatory evaluations:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
DATA_DIR=/data/cs1k-360p \
RUN_ROOT=/runs/diamond-cs1k-dust2-360p-rebuttal-v1 \
bash cs2_train/scripts/run_diamond_cs1k_dust2_rebuttal_v1.sh
```

For the deadline-driven step-20,000 amendment, evaluate the existing atomic
checkpoint from a reviewed clean analysis commit:

```bash
EXPECTED_ANALYSIS_COMMIT="$(git rev-parse HEAD)" \
PYTHON_BIN="$PWD/.venv/bin/python" \
DATA_DIR=/data/cs1k-360p \
RUN_ROOT=/runs/diamond-cs1k-dust2-360p-rebuttal-v2 \
bash cs2_train/scripts/run_diamond_cs1k_dust2_20k_endpoint.sh
```

The launcher refuses a dirty tracked worktree, verifies the two dataset hashes,
runs the target-frame action-alignment audit, and records the code commit,
config, package environment, CUDA/PyTorch environment, full `nvidia-smi -q`,
exact commands, logs, metrics, checkpoints, sample plans, and evaluator
outputs.

To retain a model-agnostic rollout archive during the analysis replay, append:

```bash
--save-rollout-archive
```

Then compute the optical-flow endpoint:

```bash
python -m cs2_train.src.evaluate_rollout_motion \
  --archive-dir /runs/diamond-cs1k-dust2-360p-rebuttal-v1/true/evaluation/midpoint-dynamics/rollout_archive \
  --bootstrap-replicates 10000
```

The scorer verifies every archive hash before loading it, records the
torch/torchvision weight provenance, retains per-sample/per-step metrics, and
uses the same round-clustered paired bootstrap convention as the pixel
endpoint.

### Reproduce Action Recoverability Ratio

The ARR probe contract is frozen separately in
`configs/temporal_arr_cs1k_dust2_v1.json`. It samples 100,000 alive-only train
windows and 10,000 alive-only validation windows before decoding, with seeds
20250725 and 20250726 respectively. Every nine-frame window becomes two
five-frame/four-transition segments sharing only the boundary frame. No test
row or generated rollout is used to train or select the probe.

The frame encoder is public DINOv2-B/14 from
`facebookresearch/dinov2@7764ea0f912e53c92e82eb78a2a1631e92725fc8`;
the expected weight SHA-256 is
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
This is an explicit reproducibility deviation from MIRA's manually gated
DINOv3-B. The backbone is frozen. A 2-layer, width-384 temporal transformer
head is selected by validation macro AP over 30 epochs.

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
DATA_DIR=/data/cs1k-360p \
ARR_ROOT=/runs/temporal-arr-cs1k-dust2-v1 \
bash cs2_train/scripts/run_temporal_action_probe_dust2_v1.sh
```

The launcher verifies both dataset hashes, refuses to overwrite an existing
output, and records the code/config/data hashes, package environment, GPU
environment, commands, sampled-window plans, feature-array hashes, validation
predictions, selected epoch, and probe checkpoint hash.

Evaluate each final rollout archive with:

```bash
python -m cs2_train.src.evaluate_rollout_arr \
  --archive-dir /runs/diamond-cs1k-dust2-360p-rebuttal-v1/true/evaluation/midpoint-dynamics/rollout_archive \
  --probe-checkpoint /runs/temporal-arr-cs1k-dust2-v1/probe/temporal_action_probe.pt \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20250728
```

For each class, ARR is generated-video AP divided by the corresponding
model-native real-video AP. AP groups exact score ties, so constant or
quantized predictions and cluster-bootstrap multiplicities cannot make the
result depend on row ordering. The scorer reports recovery of each rollout's
own conditioning actions, alignment with the true target actions, and whether
the shuffled rollout redirects toward its donor action stream. All confidence
intervals resample `round_id` and reuse the same bootstrap multiplicities for
the real ceiling and every action mode.

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
evaluation seeds across the two checkpoint arms. It also requires midpoint
and first-death evaluation to share the config, manifest, split, map, target
rate, resize, rollout/masking settings, seeds, action modes, sample count, and
round count. Window-specific sample and donor-plan hashes are retained
separately because the two window definitions intentionally choose different
source frames. It writes:

- `evaluation/rebuttal_summary.json`, with machine-readable checkpoint means,
  within-checkpoint action sensitivity, true-versus-shuffled training effects,
  the action-sensitivity difference in differences, and the complete paired
  2,500-step inline validation/rollout trajectory for both training arms,
  together with a shared cross-window contract and each window's plan hashes;
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
