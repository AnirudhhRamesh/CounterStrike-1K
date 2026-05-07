"""Deterministic evaluation for the CS2-WM DIAMOND baseline.

This script produces the metrics JSON that should back the paper table. It
does not invent FVD: if an external FVD package/model is unavailable, it writes
real/generated rollout arrays and records that FVD is pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR.parents[1], THIS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .dataset import CSDataset, collate_diamond  # noqa: E402
from .train import PRESETS, apply_action_mode, build_denoiser, load_config_defaults  # noqa: E402
from .visualize import rollout_autoregressive, rollout_one_step, save_rollout_gif  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def psnr_from_mse(mse: float) -> float:
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(max(mse, 1e-12))


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(N, T, C, H, W) in [-1, 1] -> (N, T, H, W, C) uint8."""
    return (
        frames.detach().float().cpu().clamp(-1, 1).add(1).div(2).mul(255).byte()
        .permute(0, 1, 3, 4, 2)
        .numpy()
    )


def save_mp4_batch(videos: np.ndarray, out_dir: Path, prefix: str, fps: int) -> list[str]:
    """Save (N, T, H, W, C) uint8 videos as MP4 when imageio-ffmpeg is present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, video in enumerate(videos):
        path = out_dir / f"{prefix}_{i:04d}.mp4"
        try:
            imageio.mimsave(path, list(video), fps=fps, codec="libx264", quality=8)
            paths.append(str(path))
        except Exception:  # noqa: BLE001
            break
    return paths


def build_args_from_checkpoint(ckpt: dict, cli_args: argparse.Namespace, config_payload: dict | None) -> SimpleNamespace:
    ckpt_args = dict(ckpt.get("args", {}))
    defaults = {}
    if config_payload is not None:
        defaults.update(config_payload.get("train_args", {}))
    defaults.update(ckpt_args)
    defaults["data_dir"] = str(cli_args.data_dir)
    defaults["out_dir"] = str(cli_args.out_dir)
    defaults["device"] = cli_args.device
    defaults["config"] = str(cli_args.config) if cli_args.config else None
    return SimpleNamespace(**defaults)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-batches", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=16)
    ap.add_argument("--num-denoising-steps", type=int, default=None,
                    help="Default comes from config val_denoise_steps, usually DIAMOND fast=1")
    ap.add_argument("--s-cond", type=float, default=None,
                    help="Default comes from config val_s_cond, usually DIAMOND fast=0.005")
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--action-mode", choices=["checkpoint", "true", "shuffled", "zeros"], default="checkpoint",
                    help="Action ablation used for evaluation; checkpoint uses ckpt args.action_mode.")
    ap.add_argument("--max-rollout-videos", type=int, default=64)
    ap.add_argument("--video-fps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config_payload = None
    config_defaults = {}
    if args.config is not None:
        config_payload, config_defaults = load_config_defaults(args.config)

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = build_args_from_checkpoint(ckpt, args, config_payload)
    preset = dict(PRESETS[model_args.preset])
    if getattr(model_args, "resize", None) is not None:
        preset["resize"] = tuple(model_args.resize)

    model = build_denoiser(model_args, preset, device)
    if args.weights == "ema":
        if "ema" not in ckpt:
            raise ValueError(f"{args.checkpoint} has no EMA weights")
        model.load_state_dict(ckpt["ema"]["shadow"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    denoise_steps = args.num_denoising_steps
    if denoise_steps is None:
        denoise_steps = int(config_defaults.get("val_denoise_steps", getattr(model_args, "val_denoise_steps", 1)))
    s_cond = args.s_cond
    if s_cond is None:
        s_cond = float(config_defaults.get("val_s_cond", getattr(model_args, "val_s_cond", 0.005)))
    action_mode = args.action_mode
    if action_mode == "checkpoint":
        action_mode = getattr(model_args, "action_mode", "true")

    n = model.cfg.inner_model.num_steps_conditioning
    T = n + args.rollout_steps
    ds = CSDataset(
        data_path=args.data_dir,
        split=args.split,
        T=T,
        stride=getattr(model_args, "stride", 1),
        resize=tuple(preset["resize"]),
        mode="diamond",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_diamond,
    )

    denoising_loss_sum = 0.0
    one_step_mse_sum = 0.0
    rollout_mse_sum = torch.zeros(args.rollout_steps, dtype=torch.float64)
    sample_count = 0
    batch_count = 0
    saved_real: list[np.ndarray] = []
    saved_fake: list[np.ndarray] = []
    qualitative_gif = None

    for batch in loader:
        if batch_count >= args.max_batches:
            break
        batch = batch.to(device)
        batch = apply_action_mode(batch, action_mode)
        b = batch.obs.size(0)

        loss, _ = model(batch)
        denoising_loss_sum += float(loss.item()) * b

        pred_next = rollout_one_step(
            model,
            batch,
            num_denoising_steps=denoise_steps,
            s_cond=s_cond,
        )
        gt_next = batch.obs[:, n]
        one_step_mse_sum += float(((pred_next - gt_next) ** 2).mean(dim=(1, 2, 3)).sum().item())

        pred_roll = rollout_autoregressive(
            model,
            batch,
            num_steps=args.rollout_steps,
            num_denoising_steps=denoise_steps,
            s_cond=s_cond,
        )
        gt_roll = batch.obs[:, n : n + args.rollout_steps]
        mse_step = ((pred_roll - gt_roll) ** 2).mean(dim=(0, 2, 3, 4)).double()
        rollout_mse_sum += mse_step * b

        if qualitative_gif is None:
            qualitative_gif = save_rollout_gif(
                pred_roll[: min(4, b)],
                gt_roll[: min(4, b)],
                args.out_dir / "qualitative_rollout.gif",
                fps=args.video_fps,
                max_rows=min(4, b),
            )

        remaining = args.max_rollout_videos - sum(x.shape[0] for x in saved_real)
        if remaining > 0:
            take = min(remaining, b)
            saved_fake.append(frames_to_uint8(pred_roll[:take]))
            saved_real.append(frames_to_uint8(gt_roll[:take]))

        sample_count += b
        batch_count += 1

    rollout_mse = (rollout_mse_sum / max(1, sample_count)).tolist()
    one_step_mse = one_step_mse_sum / max(1, sample_count)
    denoising_loss = denoising_loss_sum / max(1, sample_count)

    real_np = np.concatenate(saved_real, axis=0) if saved_real else np.empty((0,))
    fake_np = np.concatenate(saved_fake, axis=0) if saved_fake else np.empty((0,))
    npz_path = args.out_dir / "rollouts_for_fvd.npz"
    np.savez_compressed(npz_path, real=real_np, generated=fake_np)
    real_mp4 = save_mp4_batch(real_np, args.out_dir / "real_mp4", "real", args.video_fps)
    fake_mp4 = save_mp4_batch(fake_np, args.out_dir / "generated_mp4", "generated", args.video_fps)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "weights": args.weights,
        "config": str(args.config) if args.config else None,
        "data_dir": str(args.data_dir),
        "split": args.split,
        "num_dataset_windows": len(ds),
        "num_eval_samples": sample_count,
        "num_eval_batches": batch_count,
        "rollout_steps": args.rollout_steps,
        "num_denoising_steps": denoise_steps,
        "s_cond": s_cond,
        "action_mode": action_mode,
        "denoising_loss": denoising_loss,
        "one_step_mse": one_step_mse,
        "one_step_psnr_db": psnr_from_mse(one_step_mse),
        "rollout_mse_per_step": rollout_mse,
        "rollout_psnr_db_per_step": [psnr_from_mse(x) for x in rollout_mse],
        "rollout_mse_mean": float(np.mean(rollout_mse)) if rollout_mse else None,
        "rollout_mse_last": float(rollout_mse[-1]) if rollout_mse else None,
        "qualitative_gif": str(qualitative_gif) if qualitative_gif else None,
        "rollouts_for_fvd_npz": str(npz_path),
        "real_mp4_dir": str(args.out_dir / "real_mp4") if real_mp4 else None,
        "generated_mp4_dir": str(args.out_dir / "generated_mp4") if fake_mp4 else None,
        "fvd": None,
        "fvd_status": "pending: run cd-fvd on real_mp4_dir/generated_mp4_dir or rollouts_for_fvd.npz",
    }

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
