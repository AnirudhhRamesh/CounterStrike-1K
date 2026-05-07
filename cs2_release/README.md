# CounterStrike-1K Evaluation Suite

This is the eval-suite package for [CounterStrike-1K](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K) — the synchronized 10-POV CS2 dataset for video world modeling. The full paper baselines (Tables 5–8 and Figs 3–5) all reproduce from this directory.

It reads only public release artifacts: `manifest.parquet`, `round_index.parquet`, `sample_index_*.parquet`, and the WebDataset shards (or the unsharded reviewer sample). No raw demos, S3 control tables, Steam IDs, or HLTV identifiers required.

## Layout

```
cs2_release/
├── core/             # IO, windows, video/audio decoding, metrics, stats, tracking
├── encoders/         # Frozen video/audio feature extractors + shard runners
├── retrieval/        # Cross-POV retrieval (paper Table 6, Figs 3–5)
│   ├── pairs/        #   pair builders (basic, multi-positive, spatial)
│   └── eval_*.py     #   evaluators (basic, multi-positive, pairwise, temporal)
├── action_probe/     # Video-to-action + multi-POV probes (paper Tables 6, 7)
├── audio_probe/      # Audio-only action probe (paper Table 5)
├── global_state/     # k-POV state probes (paper Table 8)
├── corruption/       # 10-POV synchronization corruption detection
├── hygiene/          # Split-leakage / identity-column checks
├── benchmarks/       # Dataloader-format microbenchmark
├── pipelines/        # Top-level orchestrators (eval_suite)
└── tools/            # Auxiliary exporters (paper-results aggregation)
```

`python -m cs2_release` lists every entry point. `python -m cs2_release.<module> --help` shows that module's flags.

## Quickstart for researchers

`quickstart.ipynb` walks through manifest browsing → sample decoding → action overlay → 10-POV grid in seven cells.

```bash
uv run jupyter lab cs2_release/quickstart.ipynb
```

For a one-minute laptop smoke test of the full eval pipeline (RGB-histogram encoder, no GPU needed):

```bash
uv run python -m cs2_release.pipelines.eval_suite \
  --root /data/CounterStrike-1K-sample \
  --shard-root /data/CounterStrike-1K-sample \
  --resolution 360p \
  --encoder rgb_hist \
  --eval-split train \
  --max-rounds-per-split 20 \
  --max-retrieval-queries 200 \
  --corruption-epochs 3 \
  --out runs/smoke
```

The sample repo only ships `split=train` rows; `--eval-split train` points the retrieval and corruption stages at that split. For the full release, drop the flag (it defaults to `test` for paper reproduction).

## Paper-table → command map

| Paper artefact | Command |
|---|---|
| Table 5 — audio action probe | `python -m cs2_release.audio_probe.pipeline ...` |
| Table 6 — cross-POV retrieval (any synchronized) | `python -m cs2_release.pipelines.eval_suite ...` (basic retrieval stage) |
| Table 6 — cross-POV retrieval (co-located) | `python -m cs2_release.pipelines.eval_suite --run-extended ...` (spatial pairs) |
| Table 6 — video action probe | called inside `eval_suite --run-extended` |
| Table 7 — 1-POV vs 10-POV coverage | `python -m cs2_release.action_probe.coverage ...` |
| Table 7 — Off-POV miss-rate | `python -m cs2_release.action_probe.off_pov ...` |
| Table 8 — k-POV global state | `python -m cs2_release.global_state.train ...` |
| Table 9 (Appendix) — learned k-POV action | `python -m cs2_release.action_probe.train_multipov ...` |
| Table 11 (Appendix) — DIAMOND adapter | `torchrun -m cs2_train.src.train --config cs2_train/configs/diamond_csgo_lowres_paper.json ...` |
| Fig 4/5 — qualitative retrieval | `python -m cs2_release.retrieval.render_qualitative ...` |

## Reproducing the headline retrieval/probe rows

```bash
uv run python -m cs2_release.pipelines.eval_suite \
  --root /data/CounterStrike-1K \
  --shard-root /data/CounterStrike-1K \
  --resolution 360p \
  --encoder dinov2_vitb14 \
  --window-seconds 1.0 \
  --windows-per-round 3 \
  --max-rounds-per-split 250 \
  --retrieval-candidates 32 \
  --bootstrap-samples 500 \
  --run-extended \
  --frames-per-window 4 \
  --global-probe-epochs 60 \
  --out runs/paper_dinov2b
```

For the ViT-S rows, replace the encoder with `dinov2_vits14` and use `--windows-per-round 1`. Audio probe runs separately (decodes mp4 audio rather than reusing video embeddings):

```bash
uv run python -m cs2_release.audio_probe.pipeline \
  --root /data/CounterStrike-1K \
  --shard-root /data/CounterStrike-1K \
  --resolution 360p \
  --labels-backend actions_bin \
  --windows-per-round 1 \
  --max-rounds-per-split 500 \
  --audio-probe-epochs 40 \
  --bootstrap-samples 500 \
  --out runs/audio_probe
```

DIAMOND pilot (paper Table 11, Fig 6 in appendix) — runs from the dataset repo root so that `cs2_train.src.train` resolves:

```bash
torchrun --nproc_per_node=4 -m cs2_train.src.train \
  --config cs2_train/configs/diamond_csgo_lowres_paper.json \
  --data-dir /data/CounterStrike-1K \
  --out-dir runs/diamond_true \
  --action-mode true \
  --max-train-clips 20000 --max-val-clips 512 \
  --batch-size 16 --grad-acc 2 \
  --max-steps 30000 \
  --val-every 1000 --rollout-every 5000 --ckpt-every 500
```

## Sharded embedding extraction (multi-GPU)

For larger encoders, split extraction across GPUs and merge:

```bash
uv run python -m cs2_release.encoders.shard_runner \
  --root /data/CounterStrike-1K \
  --shard-root /data/CounterStrike-1K \
  --resolution 360p \
  --windows runs/paper/windows/eval_windows.parquet \
  --encoder dinov2_vitb14 \
  --devices 0 1 2 3 \
  --out runs/paper/embeddings

uv run python -m cs2_release.pipelines.eval_suite \
  --root /data/CounterStrike-1K \
  --shard-root /data/CounterStrike-1K \
  --resolution 360p \
  --encoder dinov2_vitb14 \
  --skip-embedding-extraction \
  --run-extended \
  --skip-corruption \
  --out runs/paper
```

## Outputs

Every run writes a self-contained directory with provenance metadata:

```
runs/<run_name>/
  windows/eval_windows.parquet
  embeddings/<encoder>/{embedding_index.parquet,embeddings.npz}
  retrieval/{metrics_retrieval.json,retrieval_predictions.parquet}
  spatial_retrieval/metrics_retrieval.json
  spatial_retrieval_same_time_far/metrics_retrieval.json
  spatial_retrieval_same_location_wrong_time/metrics_retrieval.json
  pairwise_spatial_probe/metrics_pairwise_spatial_probe.json
  action_probe/metrics_action_probe.json
  multipov_action_coverage/multipov_action_coverage.metrics.json
  off_pov_action_visibility/off_pov_action_visibility.metrics.json
  multipov_action_probe/metrics_multipov_action_probe.json
  global_event_probe/metrics_global_event_probe.json
  offscreen_action_probe/metrics_offscreen_action_probe.json
  temporal_alignment/metrics_temporal_alignment.json
  qualitative/spatial_retrieval_examples.png
  corruption/{checkpoint/corruption_head.pt,test/metrics_corruption.json}
  audio_probe/{metrics_audio_probe.json,audio_probe_predictions.parquet}
```

Every metrics JSON records the seed, encoder name, label-table sha256, and git commit so runs are auditable.

## W&B tracking

Every heavyweight stage takes optional W&B flags:

```bash
uv run python -m cs2_release.pipelines.eval_suite \
  ... \
  --wandb-project counterstrike-1k-evals \
  --wandb-group cs2-eval-dinov2-v1 \
  --wandb-run-name cs2-eval-dinov2-v1
```

Auth comes from `wandb login` or `WANDB_API_KEY`; no credentials are stored in this repo.

## Citation

If you use this evaluation suite, please cite the dataset:

```bibtex
@dataset{counterstrike1k2026,
  title     = {CounterStrike-1K: A Multi-Perspective Dataset of Professional Gameplay for World Modeling},
  author    = {Ramesh, Anirudhh},
  year      = {2026},
  publisher = {Hugging Face},
  version   = {1.0.0},
  url       = {https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K}
}
```
