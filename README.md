# CounterStrike-1K

**Synchronized 10-POV Counter-Strike 2 dataset for video world modeling, action-conditioned video prediction, and multi-view consistency research.** NeurIPS 2026 D&B submission.

<p align="center">
  <b>1,490.82 rendered POV-hours</b> · <b>7,347 synchronized rounds</b> · <b>73,470 POV clips</b> · <b>171.7M frames</b> · <b>7 maps</b> · <b>720p + audio</b>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CounterStrike--1K-blue"></a>
  <a href="https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K-sample"><img alt="Preview" src="https://img.shields.io/badge/%F0%9F%A4%97%20Preview-2GB-yellow"></a>
  <a href="https://github.com/AnirudhhRamesh/counterstrike1k"><img alt="Loader" src="https://img.shields.io/badge/Loader-counterstrike1k-green"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey"></a>
</p>

This repository ships the **evaluation suite** (`cs2_release/`) and **DIAMOND-style training baseline** (`cs2_train/`). The Python loader lives in [`AnirudhhRamesh/counterstrike1k`](https://github.com/AnirudhhRamesh/counterstrike1k); the dataset itself is on [Hugging Face](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K).

---

## Three ways to use this dataset

### 1. Just load some samples

Start a fresh `uv` project, add the loader, and run:

```bash
mkdir cs1k-demo && cd cs1k-demo
uv init
uv add datasets "counterstrike1k @ git+https://github.com/AnirudhhRamesh/counterstrike1k"
```

```python
from counterstrike1k import load_sample, overlay_frame

for sample in load_sample():
    overlay_frame(sample, 60).save("debug.png")
    break
```

That's it — the `counterstrike1k` package gives you `decode_sample`, `decode_actions`, `decode_state`, `overlay_frame`, `overlay_video`, and `load_sample` for the small offline preview.

### 2. Run the benchmarks (this repo)

```bash
git clone https://github.com/AnirudhhRamesh/CounterStrike-1K
cd CounterStrike-1K
uv sync --extra eval --extra notebooks
uv run jupyter lab cs2_release/quickstart.ipynb
```

The notebook walks through manifest → sample → actions/state → 10-POV grid in seven cells. `python -m cs2_release` lists every benchmark entry point.

### 3. Train an action-conditioned world model

```bash
uv sync --extra train
torchrun --nproc_per_node=4 -m cs2_train.src.train \
  --config cs2_train/configs/diamond_csgo_lowres_paper.json \
  --data-dir /data/CounterStrike-1K \
  --out-dir runs/diamond_true \
  --action-mode true \
  --max-train-clips 20000 --max-val-clips 512 \
  --batch-size 16 --grad-acc 2 --max-steps 30000
```

`cs2_train/` contains the DIAMOND-CSGO low-resolution adapter from the paper.

---

## Common recipes

**Browse the manifest** (no media downloaded):

```python
import pandas as pd
from huggingface_hub import hf_hub_download

manifest = pd.read_parquet(hf_hub_download(
    "ArnieRamesh/CounterStrike-1K", "manifest.parquet", repo_type="dataset",
))
print(len(manifest), "POV samples")
```

**Filter to one map and split**:

```python
mirage_train = manifest[(manifest["map_slug"] == "mirage") & (manifest["split"] == "train")]
```

**Pull a bandwidth-friendly subset** (10 / 50 / 100 / 500 / 1000 hours, or `dust2_100h`):

```python
ten_hours = pd.read_parquet(hf_hub_download(
    "ArnieRamesh/CounterStrike-1K", "subsets/train_10h.parquet", repo_type="dataset",
))
```

**Stream training data without downloading the full 1.3 TB**:

```python
from datasets import load_dataset
from counterstrike1k import decode_sample

shards = load_dataset(
    "ArnieRamesh/CounterStrike-1K-360-wds", split="train", streaming=True,
)
for raw in shards:
    sample = decode_sample(raw)
    # train your model on sample["video"], sample["actions"], sample["state"]
```

**Group all 10 synchronized POVs of one round**:

```python
round_id = manifest.iloc[0]["round_id"]
all_ten = manifest[manifest["round_id"] == round_id]
# 10 rows, one per pov_idx ∈ {0..9}
```

**Verify action–video alignment with the debug HUD**:

```python
from counterstrike1k import overlay_video

overlay_video(sample, "debug.mp4", max_frames=192)  # writes mp4 with WASD/score/HUD
```

---

## What's in a sample

Each WebDataset sample is one player POV across one synchronized round.

| Member | Format | Description |
|---|---|---|
| `mp4` | H.264 + AAC | 720p or 360p video at 32 FPS with synchronized stereo game audio |
| `actions.bin` | 14 B/frame | tick, `delta_pitch`, `delta_yaw`, 12-button bitmask |
| `state.bin` | 37 B/frame | view, world position, weapon, ammo, HP, armor, money, score, equipment |
| `events.json` | JSON | sparse round / kill / bomb / blind events with anonymous `pov_idx` references |
| `json` | JSON | sample metadata: ids, alignment, alive window, weapon flags, kill counts |

12 buttons: `FORWARD, BACK, LEFT, RIGHT, JUMP, DUCK, WALK, FIRE, RIGHTCLICK, RELOAD, INSPECT, USE`. Group POVs into one round via the shared `round_id`.

Full schema lives at the [`schema/` folder on HF](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K/tree/main/schema).

## Splits

| Split | POV-hours | Match-maps | Rounds | POV clips |
|---|---:|---:|---:|---:|
| train | 1,341.73 | 301 | 6,573 | 65,730 |
| val   | 74.54 | 21 | 383 | 3,830 |
| test  | 74.55 | 20 | 391 | 3,910 |
| **total** | **1,490.82** | **342** | **7,347** | **73,470** |

Split unit is the **match-map demo**, so the same match never appears in two splits. Bandwidth-friendly subsets (`train_10h`, `train_50h`, `train_100h`, `train_500h`, `train_1000h`, `train_all`, `dust2_100h`, `full_demo_eval`) are precomputed parquet files of `sample_key`s.

---

## Reproducing the paper

The full evaluation pipeline (paper Tables 5–8 and Figs 3–5) is one command:

```bash
uv run python -m cs2_release.pipelines.eval_suite \
  --root /data/CounterStrike-1K \
  --shard-root /data/CounterStrike-1K \
  --resolution 360p \
  --encoder dinov2_vitb14 \
  --windows-per-round 3 \
  --max-rounds-per-split 250 \
  --bootstrap-samples 500 \
  --run-extended \
  --frames-per-window 4 \
  --global-probe-epochs 60 \
  --out runs/paper_dinov2b
```

For the ViT-S rows: replace `--encoder dinov2_vitb14` with `dinov2_vits14` and `--windows-per-round 1`.

The audio probe (paper Table 5) decodes mp4 audio and runs separately:

```bash
uv run python -m cs2_release.audio_probe.pipeline \
  --root /data/CounterStrike-1K \
  --shard-root /data/CounterStrike-1K \
  --resolution 360p \
  --labels-backend actions_bin \
  --max-rounds-per-split 500 \
  --audio-probe-epochs 40 \
  --bootstrap-samples 500 \
  --out runs/audio_probe
```

The DIAMOND adapter pilot (paper Table 11) is a multi-GPU `torchrun` job — see the "Train an action-conditioned world model" recipe above.

`cs2_release/README.md` has the full **paper-table → command map** plus per-stage details and W&B integration.

## Smoke test (laptop-friendly, ~1 minute)

```bash
uv run python -m cs2_release.pipelines.eval_suite \
  --root /data/CounterStrike-1K-sample \
  --shard-root /data/CounterStrike-1K-sample \
  --resolution 360p \
  --encoder rgb_hist \
  --max-rounds-per-split 20 \
  --max-retrieval-queries 200 \
  --corruption-epochs 3 \
  --out runs/smoke
```

Uses the dependency-free RGB-histogram encoder against the 2 GB sample repo.

---

## Layout

```
cs2_release/             # evaluation suite (paper-aligned)
├── README.md            # paper-table → command map
├── quickstart.ipynb     # researcher walkthrough
├── core/                # IO, windows, video/audio, metrics, stats, tracking
├── encoders/            # frozen feature extractors (DINOv2, audio log-mel)
├── retrieval/           # cross-POV retrieval (Table 6, Figs 3–5)
├── action_probe/        # video-to-action + multi-POV (Tables 6, 7)
├── audio_probe/         # audio-only probe (Table 5)
├── global_state/        # k-POV state probes (Table 8)
├── corruption/          # 10-POV synchronization detection
├── hygiene/             # split leakage / identity-column checks
├── benchmarks/          # dataloader-format microbenchmark
├── pipelines/           # top-level orchestrators
└── tools/               # paper-results aggregation

cs2_train/               # DIAMOND-CSGO action-conditioned baseline (Table 11)
├── configs/             # diamond_csgo_lowres_paper.json
├── src/                 # train.py, dataset.py, action_encoder.py, ...
└── ...
```

## Repos

| Repo | What's there |
|---|---|
| **`AnirudhhRamesh/CounterStrike-1K`** (this repo) | Evaluation suite + DIAMOND baseline |
| [`AnirudhhRamesh/counterstrike1k`](https://github.com/AnirudhhRamesh/counterstrike1k) | Python loader (`decode_sample`, `overlay_frame`, …) |
| [`ArnieRamesh/CounterStrike-1K`](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K) | Dataset card, manifests, schema, subsets |
| [`ArnieRamesh/CounterStrike-1K-sample`](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K-sample) | 2 GB offline preview (one match, 16 rounds) |
| [`ArnieRamesh/CounterStrike-1K-360-wds`](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K-360-wds) | 360p WebDataset shards (~1.3 TB, recommended for training) |
| [`ArnieRamesh/CounterStrike-1K-720-wds`](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K-720-wds) | 720p WebDataset shards (~1.5 TB) |

## Citation

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

## License

CC BY-NC 4.0 to the extent of the authors' rights ([LICENSE](LICENSE)). Counter-Strike 2 and underlying game assets remain property of Valve Corporation. Raw HLTV demo files are not redistributed.

Out of scope: re-identifying players, recovering Steam IDs, profiling, ranking, anti-cheat, or surveillance use cases. Public artifacts contain no Steam IDs, online account IDs, raw HLTV identifiers, profile URLs, player names, or chat text — `pov_idx` is anonymous and stable only within a single match.
