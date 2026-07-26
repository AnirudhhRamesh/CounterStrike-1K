"""Minimal training loop for DIAMOND-style action-conditioned video diffusion
on CS2 data.

Logs to W&B if WANDB_API_KEY is set in the environment (we never read keys
from CLI flags — pass via env). Falls back to plain stdout otherwise.

Usage:
    export WANDB_API_KEY=...                # from your shell, NOT committed
    uv run python -m src.train \\
        --data-dir /opt/dlami/nvme/cs2-data \\
        --out-dir runs/dev01 \\
        --preset small               # or `full` for DIAMOND-CSGO size
        --max-steps 20000 \\
        --val-every 250

Loss should drop from ~1.0 to ~0.4-0.5 within a few hundred steps even on
the small preset; meaningful frame predictions appear after several
thousand steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Required by torch.use_deterministic_algorithms(True) for deterministic CUDA
# matrix multiplications on CUDA 10.2 and newer.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# repo-relative imports
THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR.parents[1], THIS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .action_encoder import NUM_ACTIONS
from .dataset import CSDataset, collate_diamond
from .diamond import (
    Denoiser,
    DenoiserConfig,
    InnerModelConfig,
    SigmaDistributionConfig,
)
from .review_publisher import ReviewPublisher
from .visualize import run_rollout_validation, run_validation

LOG = logging.getLogger("cs2_train")


def setup_logging(out_dir: Path, *, rank: int = 0) -> None:
    """Set up python logging to stdout + <out_dir>/train.log.

    File handler is append-mode so resumes accumulate. Stdout stays unbuffered
    so tmux/CI follow-along works.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    LOG.addHandler(sh)
    log_name = "train.log" if rank == 0 else f"train_rank{rank}.log"
    fh = logging.FileHandler(out_dir / log_name, mode="a")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)


def init_distributed(args) -> tuple[torch.device, int, int, int, bool]:
    """Initialize torchrun/DDP when WORLD_SIZE is set.

    Returns (device, rank, local_rank, world_size, is_main).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA in this trainer")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    return device, rank, local_rank, world_size, rank == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


# ---------------------------------------------------------------------------
# Lightweight EMA over parameters (DIAMOND uses 0.999 in CSGO config).
# ---------------------------------------------------------------------------


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:  # buffers like int counters: just copy.
                s.copy_(v)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        self.shadow = sd["shadow"]

    def swap_into(self, model: torch.nn.Module) -> dict:
        """Swap EMA weights into model, returning a saved-original dict."""
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return original

    def restore(self, model: torch.nn.Module, original: dict) -> None:
        model.load_state_dict(original, strict=True)


# ---------------------------------------------------------------------------
# Model presets — `small` for dev iteration, `full` matches DIAMOND-CSGO paper.
# ---------------------------------------------------------------------------

PRESETS = {
    "small": {
        "cond_channels": 256,
        "base_channels": 64,  # channels = [64, 128, 256, 512]
        "depths": [2, 2, 2, 2],
        "attn_depths": [0, 0, 1, 1],
        "resize": (36, 64),
    },
    "medium": {
        "cond_channels": 1024,
        "base_channels": 96,  # channels = [96, 192, 384, 768]
        "depths": [2, 2, 2, 2],
        "attn_depths": [0, 0, 1, 1],
        "resize": (36, 64),
    },
    "full": {
        # DIAMOND-CSGO low-res denoiser (config/agent/csgo.yaml).
        "cond_channels": 2048,
        "base_channels": 128,  # channels = [128, 256, 512, 1024]
        "depths": [2, 2, 2, 2],
        "attn_depths": [0, 0, 1, 1],
        "resize": (36, 64),
    },
}


def load_config_defaults(path: Path) -> tuple[dict, dict]:
    """Load a checked-in JSON baseline config and return argparse defaults.

    We intentionally use JSON instead of YAML to avoid adding a dependency to
    the training path. CLI flags still win over these defaults.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = {
        k.replace("-", "_"): v for k, v in payload.get("train_args", {}).items()
    }
    return payload, defaults


def build_denoiser(args, preset: dict, device: torch.device) -> Denoiser:
    inner = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=args.cond_frames,
        cond_channels=preset["cond_channels"],
        depths=list(preset["depths"]),
        channels=[preset["base_channels"] * m for m in (1, 2, 4, 8)],
        attn_depths=list(preset["attn_depths"]),
        num_actions=NUM_ACTIONS,
        is_upsampler=False,
    )
    cfg = DenoiserConfig(
        inner_model=inner,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
        noise_previous_obs=True,
        upsampling_factor=None,
    )
    model = Denoiser(cfg).to(device)
    model.setup_training(
        SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20)
    )
    return model


def lr_lambda(step: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / float(max(1, warmup))
    return 1.0


def save_checkpoint(
    path: Path, *, model, optim, sched, step, best_val_mse, args, ema=None
):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "best_val_mse": best_val_mse,
        "args": vars(args),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on POSIX


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def sequence_derangement(
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    labels: list[str] | None = None,
) -> torch.Tensor:
    """Return a random cyclic derangement for a batch.

    A non-zero cyclic shift has no fixed points and, unlike a flattened
    frame-level permutation, preserves each donor clip's temporal action
    sequence and marginal action statistics.
    """

    if batch_size < 2:
        raise ValueError("shuffled actions require batch_size >= 2")
    valid_shifts = list(range(1, batch_size))
    if labels is not None:
        if len(labels) != batch_size:
            raise ValueError(f"expected {batch_size} shuffle labels, got {len(labels)}")
        valid_shifts = [
            shift
            for shift in valid_shifts
            if all(
                labels[i] != labels[(i + shift) % batch_size] for i in range(batch_size)
            )
        ]
        if not valid_shifts:
            raise ValueError(
                "could not construct a cross-round action derangement for this batch"
            )
    choice = int(torch.randint(len(valid_shifts), (1,), generator=generator).item())
    shift = valid_shifts[choice]
    return (torch.arange(batch_size, device=device) + shift) % batch_size


def apply_action_mode(
    batch,
    mode: str,
    *,
    generator: torch.Generator | None = None,
):
    """Apply action ablations without changing the DIAMOND batch contract."""

    if mode == "true":
        return batch
    if mode == "zeros":
        act = torch.zeros_like(batch.act)
        info = [{**item, "action_source_batch_idx": None} for item in batch.info]
    elif mode == "shuffled":
        labels = [
            f"{item.get('match_id', '')}/{item.get('round_id', '')}"
            for item in batch.info
        ]
        perm = sequence_derangement(
            batch.act.shape[0],
            generator=generator,
            device=batch.act.device,
            labels=labels,
        )
        act = batch.act[perm]
        perm_cpu = perm.cpu().tolist()
        info = [
            {
                **item,
                "action_source_batch_idx": int(source_idx),
                "action_source_sample_key": batch.info[source_idx].get(
                    "sample_key", ""
                ),
            }
            for item, source_idx in zip(batch.info, perm_cpu, strict=True)
        ]
    else:
        raise ValueError(f"unknown action_mode={mode!r}")
    return batch.__class__(
        obs=batch.obs,
        act=act,
        rew=batch.rew,
        end=batch.end,
        trunc=batch.trunc,
        mask_padding=batch.mask_padding,
        info=info,
        segment_ids=batch.segment_ids,
    )


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file with baseline defaults; CLI flags override it",
    )
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(description=__doc__, parents=[pre])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)

    # Window
    ap.add_argument(
        "--cond-frames", type=int, default=4, help="DIAMOND num_steps_conditioning (n)"
    )
    ap.add_argument("--num-autoregressive-steps", type=int, default=1)
    ap.add_argument(
        "--target-fps",
        type=int,
        default=8,
        help="Emitted video/action rate; must divide the locked 32-fps source.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Deprecated explicit source-frame stride; must equal 32/target_fps.",
    )
    ap.add_argument(
        "--train-window-mode",
        choices=["sliding", "midpoint", "first-death"],
        default="sliding",
    )
    ap.add_argument(
        "--val-window-mode",
        choices=["midpoint", "first-death"],
        default="midpoint",
    )
    ap.add_argument(
        "--action-mode",
        choices=["true", "shuffled", "zeros"],
        default="true",
        help="Action conditioning ablation. Use true for the main baseline.",
    )
    ap.add_argument("--action-shuffle-seed", type=int, default=90_001)

    # Model preset
    ap.add_argument("--preset", choices=list(PRESETS), default="small")
    ap.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        help="Override preset's resize (H W)",
    )

    # Optim
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-acc", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lr-warmup", type=int, default=100)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--mixed-precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument(
        "--scale-loss-by-grad-acc",
        action="store_true",
        default=True,
        help="Average accumulated gradients. Disable to match upstream DIAMOND trainer.",
    )
    ap.add_argument(
        "--no-scale-loss-by-grad-acc",
        dest="scale_loss_by_grad_acc",
        action="store_false",
    )
    ap.add_argument(
        "--ddp-preserve-loss-scale",
        action="store_true",
        default=True,
        help="When using DDP and unscaled grad accumulation, multiply local loss by world size so DDP averaging preserves the upstream single-process loss scale.",
    )
    ap.add_argument(
        "--no-ddp-preserve-loss-scale",
        dest="ddp_preserve_loss_scale",
        action="store_false",
    )

    # Loader
    ap.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Legacy manifest JSON or public release manifest.parquet.",
    )
    ap.add_argument(
        "--map-slug",
        default="dust2",
        help="Map filter. Rebuttal configs pin this to dust2.",
    )
    ap.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    ap.add_argument(
        "--shard-root",
        type=Path,
        default=None,
        help="Local shard root for public release WDS shards; defaults to --data-dir.",
    )
    ap.add_argument(
        "--subset",
        default=None,
        help="Optional subsets/<name>.parquet filter for public release roots.",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Materialized MP4/action cache for release WDS samples.",
    )
    ap.add_argument(
        "--max-train-clips",
        type=int,
        default=None,
        help="Optional deterministic clip cap for release-root smoke runs.",
    )
    ap.add_argument(
        "--max-val-clips",
        type=int,
        default=None,
        help="Optional deterministic validation clip cap for release-root smoke runs.",
    )
    ap.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Verify WDS member hashes while materializing release samples.",
    )
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--persistent-workers", action="store_true", default=True)
    ap.add_argument(
        "--no-persistent-workers", dest="persistent_workers", action="store_false"
    )

    # Logging / eval
    ap.add_argument(
        "--log-every", type=int, default=20, help="Step-level metrics cadence"
    )
    ap.add_argument(
        "--val-every",
        type=int,
        default=500,
        help="Single-step val (PSNR + grid PNG) every N steps",
    )
    ap.add_argument(
        "--rollout-every",
        type=int,
        default=2000,
        help="Multi-step autoregressive rollout (saves GIF + uploads to W&B) every N steps",
    )
    ap.add_argument(
        "--rollout-steps",
        type=int,
        default=16,
        help="Number of frames to roll out autoregressively for the GIF",
    )
    ap.add_argument("--val-batch-size", type=int, default=4)
    ap.add_argument("--val-rows", type=int, default=4)
    ap.add_argument("--val-denoise-steps", type=int, default=10)
    ap.add_argument(
        "--val-s-cond",
        type=float,
        default=0.0,
        help="DIAMOND sampler s_cond used for validation rollouts",
    )
    ap.add_argument(
        "--use-ema-for-val",
        action="store_true",
        default=True,
        help="Run validation with EMA weights swapped in (and restored after).",
    )
    ap.add_argument(
        "--no-use-ema-for-val", dest="use_ema_for_val", action="store_false"
    )

    # EMA
    ap.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay; 0 to disable EMA."
    )

    # Checkpointing
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=10000,
        help="Save step_NNNNNN.pt every N (0 to disable)",
    )
    ap.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Maintain best.pt using the fixed true-action validation MSE.",
    )
    ap.add_argument(
        "--save-latest",
        action="store_true",
        default=True,
        help="Always update latest.pt after each val cycle",
    )
    ap.add_argument("--no-save-latest", dest="save_latest", action="store_false")
    ap.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write latest.pt at normal completion (disable only for smoke tests).",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from this checkpoint (or auto-resume from <out-dir>/latest.pt if it exists)",
    )

    # W&B
    ap.add_argument("--wandb-project", default="cs2-wm")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument(
        "--wandb-mode", default="online", choices=["online", "offline", "disabled"]
    )

    # Private live review publishing
    ap.add_argument(
        "--review-s3-uri",
        default=None,
        help="Private S3 root for paired validation/rollout artifacts.",
    )
    ap.add_argument("--review-run-id", default=None)
    ap.add_argument("--review-arm", default=None)
    ap.add_argument("--review-url-expires", type=int, default=604_800)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic PyTorch/CUDA algorithms for reproducible runs.",
    )

    config_payload = None
    if pre_args.config is not None:
        config_payload, defaults = load_config_defaults(pre_args.config)
        ap.set_defaults(**defaults)

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device, rank, local_rank, world_size, is_main = init_distributed(args)
    val_dir = args.out_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = args.out_dir / "rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.out_dir, rank=rank)
    metrics_path = args.out_dir / "metrics.jsonl"

    review_publisher = None
    if is_main and args.review_s3_uri:
        review_publisher = ReviewPublisher(
            s3_uri=args.review_s3_uri,
            run_id=args.review_run_id or args.out_dir.name,
            arm=args.review_arm or args.action_mode,
            local_dir=args.out_dir,
            expires_seconds=args.review_url_expires,
        )
        LOG.info(
            "Private review publishing enabled: "
            f"{args.review_s3_uri.rstrip('/')}/{review_publisher.run_id}"
        )
        review_publisher.mark_running()

    preset = dict(PRESETS[args.preset])
    if args.resize is not None:
        preset["resize"] = tuple(args.resize)

    if is_main:
        (args.out_dir / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "preset_resolved": preset,
                    "baseline_config": config_payload,
                    "distributed": {
                        "world_size": world_size,
                        "rank": rank,
                        "local_rank": local_rank,
                        "backend": "nccl" if world_size > 1 else None,
                    },
                    "global_effective_batch": args.batch_size
                    * args.grad_acc
                    * world_size,
                },
                indent=2,
                default=str,
            )
        )
    barrier()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    action_generator = torch.Generator().manual_seed(args.action_shuffle_seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
    # Need at least cond_frames + rollout_steps frames per window for the
    # autoregressive rollout val to use ground-truth actions throughout.
    train_T = args.cond_frames + 1 + args.num_autoregressive_steps
    val_T = max(train_T, args.cond_frames + max(args.rollout_steps, 1))
    LOG.info(
        f"distributed: rank={rank} local_rank={local_rank} world_size={world_size} device={device}"
    )
    LOG.info(
        f"window sizes: train_T={train_T}, val_T={val_T} (rollout_steps={args.rollout_steps})"
    )

    # ---- W&B init -----------------------------------------------------------
    # Auth precedence (wandb's own): WANDB_API_KEY env > ~/.netrc > prompt.
    # We don't gate on the env var so users with `wandb login` (~/.netrc) work.
    use_wandb = False
    wandb_run = None
    if is_main and args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.out_dir.name,
                config={
                    **vars(args),
                    "preset_resolved": preset,
                    "num_actions": NUM_ACTIONS,
                    "train_T": train_T,
                    "val_T": val_T,
                },
                dir=str(args.out_dir),
                mode=args.wandb_mode,
            )
            use_wandb = True
            LOG.info(
                f"W&B: logging to {wandb_run.url if wandb_run and wandb_run.url else '(offline)'}"
            )
        except ImportError:
            LOG.warning("W&B: wandb not installed — skipping (uv add wandb)")
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"W&B: init failed ({e}) — running without it")

    # ---- Datasets -----------------------------------------------------------
    LOG.info(
        f"Building dataset (train_T={train_T}, val_T={val_T}, resize={preset['resize']})..."
    )
    train_ds = CSDataset(
        data_path=args.data_dir,
        split="train",
        T=train_T,
        stride=args.stride,
        target_fps=args.target_fps,
        window_mode=args.train_window_mode,
        map_slug=args.map_slug,
        resize=tuple(preset["resize"]),
        manifest_name=args.manifest_name,
        mode="diamond",
        resolution=args.resolution,
        shard_root=args.shard_root,
        subset=args.subset,
        cache_dir=args.cache_dir,
        max_clips=args.max_train_clips,
        seed=args.seed,
        verify_sha256=args.verify_sha256,
    )
    val_ds = CSDataset(
        data_path=args.data_dir,
        split="val",
        T=val_T,
        stride=args.stride,
        target_fps=args.target_fps,
        window_mode=args.val_window_mode,
        map_slug=args.map_slug,
        resize=tuple(preset["resize"]),
        manifest_name=args.manifest_name,
        mode="diamond",
        resolution=args.resolution,
        shard_root=args.shard_root,
        subset=args.subset,
        cache_dir=args.cache_dir,
        max_clips=args.max_val_clips,
        seed=args.seed + 1,
        verify_sha256=args.verify_sha256,
    )
    LOG.info(f"  train: {len(train_ds)} windows / {len(train_ds.samples)} clips")
    LOG.info(f"  val:   {len(val_ds)} windows / {len(val_ds.samples)} clips")

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_diamond,
    )

    # Fixed val batch — same windows every time, so the image grid is comparable across steps.
    if is_main:
        rng = torch.Generator().manual_seed(123)
        first_index_by_round: dict[str, int] = {}
        for index in range(len(val_ds)):
            clip, _start, _frame_ids = val_ds._resolve_window(index)
            first_index_by_round.setdefault(str(clip.get("round_id", "")), index)
        round_indices = list(first_index_by_round.values())
        if len(round_indices) < args.val_batch_size:
            raise ValueError(
                f"validation needs {args.val_batch_size} distinct rounds, "
                f"found {len(round_indices)}"
            )
        order = torch.randperm(len(round_indices), generator=rng)[: args.val_batch_size]
        val_indices = [round_indices[int(i)] for i in order]
        LOG.info(f"Val indices (fixed): {val_indices}")
        fixed_val_batch = collate_diamond([val_ds[i] for i in val_indices])
        # Inline validation is a convergence diagnostic shared by both arms.
        # The confirmatory evaluator performs paired true/shuffled inference.
        fixed_val_batch = apply_action_mode(fixed_val_batch, "true")
        fixed_val_batches = {
            "true": fixed_val_batch,
            "shuffled": apply_action_mode(
                fixed_val_batch,
                "shuffled",
                generator=torch.Generator().manual_seed(args.action_shuffle_seed),
            ),
        }
    else:
        fixed_val_batch = None
        fixed_val_batches = {}

    # ---- Model + optim ------------------------------------------------------
    LOG.info(f"Building Denoiser preset={args.preset}: {preset}")
    raw_model = build_denoiser(args, preset, device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    LOG.info(f"  parameters: {n_params / 1e6:.1f}M")

    autocast_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    autocast_enabled = args.mixed_precision == "bf16"

    # ---- Resume -------------------------------------------------------------
    start_step = 0
    best_val_mse = float("inf")
    resume_path: Path | None = args.resume
    auto_resume = args.out_dir / "latest.pt"
    if resume_path is None and auto_resume.exists():
        resume_path = auto_resume
        LOG.info(f"Auto-resuming from {resume_path}")
    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        best_val_mse = float(ckpt.get("best_val_mse", float("inf")))
        LOG.info(f"  resumed at step {start_step}, best_val_mse={best_val_mse:.4f}")

    model: torch.nn.Module = raw_model
    if world_size > 1:
        model = DistributedDataParallel(
            raw_model, device_ids=[local_rank], output_device=local_rank
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_lambda(s, args.lr_warmup)
    )
    if resume_path and resume_path.exists():
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])

    ema: EMA | None = (
        EMA(raw_model, decay=args.ema_decay) if args.ema_decay > 0 else None
    )
    if ema is not None:
        if resume_path and resume_path.exists() and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            LOG.info("  resumed EMA shadow weights from checkpoint")
        elif resume_path and resume_path.exists():
            ema.shadow = {
                k: v.detach().clone() for k, v in raw_model.state_dict().items()
            }
            LOG.info("  no EMA in checkpoint — re-seeded shadow from model")
        LOG.info(f"  EMA enabled (decay={args.ema_decay})")

    LOG.info(
        f"Training: max_steps={args.max_steps}, effective_batch={args.batch_size * args.grad_acc}, "
        f"global_effective_batch={args.batch_size * args.grad_acc * world_size}, "
        f"autocast={args.mixed_precision}, val_every={args.val_every}, rollout_every={args.rollout_every}"
    )

    if is_main and use_wandb:
        import wandb

        wandb.run.summary["model/n_params"] = n_params
        wandb.run.summary["model/preset"] = args.preset
        wandb.run.summary["train/global_effective_batch"] = (
            args.batch_size * args.grad_acc * world_size
        )

    # ---- Training loop ------------------------------------------------------
    step = start_step
    micro_step = 0
    optim.zero_grad(set_to_none=True)
    log_loss = 0.0
    log_n = 0
    t_log = time.perf_counter()
    samples_since_log = 0

    model.train()
    data_iter = iter(train_loader)
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = batch.to(device)
        batch = apply_action_mode(
            batch,
            args.action_mode,
            generator=action_generator,
        )
        sync_grad = ((micro_step + 1) % args.grad_acc) == 0
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not sync_grad
            else nullcontext()
        )
        with sync_context:
            with torch.amp.autocast(
                device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled
            ):
                loss, _logs = model(batch)

            if args.scale_loss_by_grad_acc:
                backward_loss = loss / args.grad_acc
            elif world_size > 1 and args.ddp_preserve_loss_scale:
                backward_loss = loss * world_size
            else:
                backward_loss = loss
            backward_loss.backward()
        log_loss += loss.detach().item()
        log_n += 1
        samples_since_log += batch.obs.size(0)
        micro_step += 1

        if not sync_grad:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optim.step()
        sched.step()
        if ema is not None:
            ema.update(raw_model)
        optim.zero_grad(set_to_none=True)
        step += 1

        # ---- Step-level logging ---------------------------------------------
        if step % args.log_every == 0:
            dt = time.perf_counter() - t_log
            lr_now = sched.get_last_lr()[0]
            samples_per_s = samples_since_log * world_size / dt
            mean_loss = log_loss / log_n
            mem_alloc = (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            if is_main:
                LOG.info(
                    f"step {step:6d}  loss {mean_loss:.4f}  "
                    f"grad {float(grad_norm):.2f}  lr {lr_now:.2e}  "
                    f"sps {samples_per_s:6.1f}  "
                    f"({dt * 1000 / log_n:.1f} ms/microstep)  "
                    f"mem {mem_alloc:.1f} GiB"
                )
                append_jsonl(
                    metrics_path,
                    {
                        "kind": "train",
                        "step": step,
                        "action_mode": args.action_mode,
                        "loss": mean_loss,
                        "grad_norm": float(grad_norm),
                        "lr": lr_now,
                        "samples_per_sec": samples_per_s,
                        "microstep_ms": dt * 1000 / log_n,
                        "cuda_max_mem_gib": mem_alloc,
                    },
                )
            if is_main and use_wandb:
                import wandb

                wandb.log(
                    {
                        "train/loss": mean_loss,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": lr_now,
                        "train/samples_per_sec": samples_per_s,
                        "train/microstep_ms": dt * 1000 / log_n,
                        "train/cuda_max_mem_gib": mem_alloc,
                    },
                    step=step,
                )
            log_loss = 0.0
            log_n = 0
            samples_since_log = 0
            t_log = time.perf_counter()

        # ---- Validation ------------------------------------------------------
        if is_main and args.val_every and step % args.val_every == 0:
            # Optionally swap EMA weights in for evaluation, then restore.
            saved = None
            # Only use EMA weights once the shadow has accumulated enough
            # history (otherwise validation looks random for the first ~1000 steps).
            if ema is not None and args.use_ema_for_val and step >= 1000:
                saved = ema.swap_into(raw_model)
            try:
                metrics_by_mode = {}
                fork_devices = [device.index or 0] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=fork_devices):
                    for mode, eval_batch in fixed_val_batches.items():
                        eval_seed = args.seed * 1_000_003 + step * 101 + 1
                        torch.manual_seed(eval_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(eval_seed)
                        metrics_by_mode[mode] = run_validation(
                            denoiser=raw_model,
                            val_batch=eval_batch,
                            out_dir=val_dir,
                            step=step,
                            num_denoising_steps=args.val_denoise_steps,
                            s_cond=args.val_s_cond,
                            max_rows=args.val_rows,
                            save_png=True,
                            tag=mode,
                        )
            finally:
                if saved is not None:
                    ema.restore(raw_model, saved)

            metrics = metrics_by_mode["true"]
            shuffled_metrics = metrics_by_mode["shuffled"]
            LOG.info(
                f"  [val] step {step}  true_mse={metrics['val_mse']:.4f}  "
                f"shuffled_mse={shuffled_metrics['val_mse']:.4f}  "
                f"delta={shuffled_metrics['val_mse'] - metrics['val_mse']:+.4f}"
            )
            append_jsonl(
                metrics_path,
                {
                    "kind": "validation",
                    "step": step,
                    "weights": "ema" if saved is not None else "raw",
                    "true": metrics,
                    "shuffled": shuffled_metrics,
                    "shuffled_minus_true_mse": (
                        shuffled_metrics["val_mse"] - metrics["val_mse"]
                    ),
                },
            )
            if use_wandb:
                import wandb

                log_payload = {
                    "val/mse": metrics["val_mse"],
                    "val/psnr_db": metrics["val_psnr_db"],
                    "val/shuffled_mse": shuffled_metrics["val_mse"],
                    "val/shuffled_psnr_db": shuffled_metrics["val_psnr_db"],
                    "val/shuffled_minus_true_mse": (
                        shuffled_metrics["val_mse"] - metrics["val_mse"]
                    ),
                }
                if metrics["val_image_path"]:
                    log_payload["val/true_grid"] = wandb.Image(
                        metrics["val_image_path"]
                    )
                if shuffled_metrics["val_image_path"]:
                    log_payload["val/shuffled_grid"] = wandb.Image(
                        shuffled_metrics["val_image_path"]
                    )
                wandb.log(log_payload, step=step)

            if review_publisher is not None:
                try:
                    review_publisher.publish(
                        step=step,
                        kind="one_step",
                        artifacts={
                            "true_grid": Path(metrics["val_image_path"]),
                            "shuffled_grid": Path(shuffled_metrics["val_image_path"]),
                        },
                        metrics={
                            "true_mse": metrics["val_mse"],
                            "true_psnr_db": metrics["val_psnr_db"],
                            "shuffled_mse": shuffled_metrics["val_mse"],
                            "shuffled_psnr_db": shuffled_metrics["val_psnr_db"],
                            "shuffled_minus_true_mse": (
                                shuffled_metrics["val_mse"] - metrics["val_mse"]
                            ),
                        },
                    )
                    LOG.info(
                        "  [review] one-step artifacts published; "
                        "signed index stored in review_index_url.txt"
                    )
                except Exception:
                    LOG.exception(
                        "  [review] one-step publish failed; training continues"
                    )

            improved = metrics["val_mse"] < best_val_mse
            if improved:
                best_val_mse = metrics["val_mse"]
                if args.save_best:
                    save_checkpoint(
                        args.out_dir / "best.pt",
                        model=raw_model,
                        optim=optim,
                        sched=sched,
                        step=step,
                        best_val_mse=best_val_mse,
                        args=args,
                        ema=ema,
                    )
                    LOG.info("  -> new best val mse, saved best.pt")

            if args.save_latest:
                save_checkpoint(
                    args.out_dir / "latest.pt",
                    model=raw_model,
                    optim=optim,
                    sched=sched,
                    step=step,
                    best_val_mse=best_val_mse,
                    args=args,
                    ema=ema,
                )

        # ---- Multi-step rollout (heavier, runs less often) -------------------
        if is_main and args.rollout_every and step % args.rollout_every == 0:
            saved = None
            # Only use EMA weights once the shadow has accumulated enough
            # history (otherwise validation looks random for the first ~1000 steps).
            if ema is not None and args.use_ema_for_val and step >= 1000:
                saved = ema.swap_into(raw_model)
            try:
                t0 = time.perf_counter()
                rolls_by_mode = {}
                fork_devices = [device.index or 0] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=fork_devices):
                    for mode, eval_batch in fixed_val_batches.items():
                        eval_seed = args.seed * 1_000_003 + step * 101 + 2
                        torch.manual_seed(eval_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(eval_seed)
                        rolls_by_mode[mode] = run_rollout_validation(
                            denoiser=raw_model,
                            val_batch=eval_batch,
                            out_dir=rollout_dir,
                            step=step,
                            num_steps=args.rollout_steps,
                            num_denoising_steps=args.val_denoise_steps,
                            s_cond=args.val_s_cond,
                            max_rows=args.val_rows,
                            fps=args.target_fps,
                            tag=mode,
                        )
                roll_dt = time.perf_counter() - t0
            finally:
                if saved is not None:
                    ema.restore(raw_model, saved)

            roll = rolls_by_mode["true"]
            shuffled_roll = rolls_by_mode["shuffled"]
            true_mean = float(
                sum(roll["rollout_mse_per_step"]) / len(roll["rollout_mse_per_step"])
            )
            shuffled_mean = float(
                sum(shuffled_roll["rollout_mse_per_step"])
                / len(shuffled_roll["rollout_mse_per_step"])
            )
            LOG.info(
                f"  [rollout] step {step}  steps={roll['rollout_steps']}  "
                f"true_mean={true_mean:.4f} shuffled_mean={shuffled_mean:.4f} "
                f"delta={shuffled_mean - true_mean:+.4f}  ({roll_dt:.1f}s)"
            )
            append_jsonl(
                metrics_path,
                {
                    "kind": "rollout",
                    "step": step,
                    "weights": "ema" if saved is not None else "raw",
                    "true": roll,
                    "shuffled": shuffled_roll,
                    "true_mse_mean": true_mean,
                    "shuffled_mse_mean": shuffled_mean,
                    "shuffled_minus_true_mse_mean": shuffled_mean - true_mean,
                    "wallclock_s": roll_dt,
                },
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "rollout/true_gif": wandb.Video(
                            roll["rollout_gif_path"], fps=args.target_fps, format="gif"
                        ),
                        "rollout/shuffled_gif": wandb.Video(
                            shuffled_roll["rollout_gif_path"],
                            fps=args.target_fps,
                            format="gif",
                        ),
                        "rollout/mse_step0": roll["rollout_mse_per_step"][0],
                        "rollout/mse_last": roll["rollout_mse_per_step"][-1],
                        "rollout/mse_mean": true_mean,
                        "rollout/shuffled_mse_mean": shuffled_mean,
                        "rollout/shuffled_minus_true_mse_mean": shuffled_mean
                        - true_mean,
                        "rollout/mse_per_step": wandb.Histogram(
                            roll["rollout_mse_per_step"]
                        ),
                        "rollout/wallclock_s": roll_dt,
                    },
                    step=step,
                )
            if review_publisher is not None:
                try:
                    review_publisher.publish(
                        step=step,
                        kind="rollout",
                        artifacts={
                            "true_rollout": Path(roll["rollout_gif_path"]),
                            "shuffled_rollout": Path(shuffled_roll["rollout_gif_path"]),
                        },
                        metrics={
                            "true_mse_per_step": roll["rollout_mse_per_step"],
                            "shuffled_mse_per_step": shuffled_roll[
                                "rollout_mse_per_step"
                            ],
                            "true_mse_mean": true_mean,
                            "shuffled_mse_mean": shuffled_mean,
                            "shuffled_minus_true_mse_mean": shuffled_mean - true_mean,
                        },
                    )
                    LOG.info(
                        "  [review] rollout artifacts published; "
                        "signed index stored in review_index_url.txt"
                    )
                except Exception:
                    LOG.exception(
                        "  [review] rollout publish failed; training continues"
                    )

        # ---- Periodic numbered checkpoint ------------------------------------
        if is_main and args.ckpt_every and step % args.ckpt_every == 0:
            ckpt_path = args.out_dir / f"step_{step:07d}.pt"
            save_checkpoint(
                ckpt_path,
                model=raw_model,
                optim=optim,
                sched=sched,
                step=step,
                best_val_mse=best_val_mse,
                args=args,
                ema=ema,
            )
            LOG.info(f"  -> saved {ckpt_path}")

    # Final
    if is_main:
        if args.save_final:
            save_checkpoint(
                args.out_dir / "latest.pt",
                model=raw_model,
                optim=optim,
                sched=sched,
                step=step,
                best_val_mse=best_val_mse,
                args=args,
                ema=ema,
            )
        LOG.info(f"Done. step={step}, best_val_mse={best_val_mse:.4f}")
        if review_publisher is not None:
            try:
                review_publisher.mark_complete(final_step=step)
                LOG.info(
                    "  [review] arm marked complete; "
                    "signed index stored in review_index_url.txt"
                )
            except Exception:
                LOG.exception("  [review] completion publish failed")
    barrier()
    if is_main and use_wandb:
        import wandb

        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
