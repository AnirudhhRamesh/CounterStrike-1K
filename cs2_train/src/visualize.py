"""Validation-time visualization: roll out the denoiser on a val batch and
render a grid PNG of (conditioning frames | predicted next frame | ground truth | |diff|).

Plus an autoregressive multi-step rollout that saves a side-by-side GIF
(predicted vs. ground truth) for qualitative inspection.

Used both:
  - inline during training (called from train.py every N steps)
  - offline by `notebooks/inspect.ipynb`, which just loads the saved PNGs/GIFs
    instead of regenerating (saves notebook re-run time + storage).
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from .diamond import (
    Batch,
    Denoiser,
    DiffusionSampler,
    DiffusionSamplerConfig,
)


@torch.no_grad()
def rollout_one_step(
    denoiser: Denoiser,
    batch: Batch,
    num_denoising_steps: int = 10,
    sigma_min: float = 2e-3,
    sigma_max: float = 5.0,
    s_cond: float = 0.0,
) -> torch.Tensor:
    """Predict the (n+1)-th frame from the first n conditioning frames.

    Args:
        denoiser: trained Denoiser (in eval mode).
        batch: a `Batch` produced by collate_diamond. We use the first
            `num_steps_conditioning` frames + their actions as conditioning;
            we predict the next one.
        num_denoising_steps: Heun-ish number of integrator steps.

    Returns:
        Predicted next frame, shape (b, c, h, w), in [-1, 1].
    """
    n = denoiser.cfg.inner_model.num_steps_conditioning
    sampler = DiffusionSampler(
        denoiser,
        DiffusionSamplerConfig(
            num_steps_denoising=num_denoising_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=7,
            order=1,
            s_churn=0,
            s_cond=s_cond,
        ),
    )

    prev_obs = batch.obs[:, :n].contiguous()        # (b, n, c, h, w)
    prev_act = batch.act[:, :n].contiguous()        # (b, n, num_actions)
    pred, _trajectory = sampler.sample(prev_obs, prev_act)
    return pred  # (b, c, h, w) in [-1, 1]


def _to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """(c, h, w) in [-1, 1] -> (h, w, c) uint8."""
    img = t.detach().float().cpu().clamp(-1, 1).add(1).div(2).mul(255).byte()
    return img.permute(1, 2, 0).numpy()


def make_grid(
    batch: Batch,
    pred_next: torch.Tensor,
    n_cond: int,
    max_rows: int = 4,
    title: str | None = None,
) -> plt.Figure:
    """Build a (rows = samples, cols = n_cond + 3) image grid.
    Last 3 cols are: predicted next, ground-truth next, |diff|."""
    b = min(max_rows, batch.obs.size(0))
    obs = batch.obs[:b]                              # (b, T, c, h, w)
    gt_next = obs[:, n_cond]                         # (b, c, h, w)
    diff = (pred_next[:b] - gt_next).abs() / 2.0     # in [0, 1] roughly

    n_cols = n_cond + 3
    fig, axes = plt.subplots(b, n_cols, figsize=(2.0 * n_cols, 2.0 * b), squeeze=False)
    for r in range(b):
        for c in range(n_cond):
            axes[r, c].imshow(_to_uint8_image(obs[r, c]))
            axes[r, c].set_title(f"cond {c}", fontsize=8) if r == 0 else None
            axes[r, c].axis("off")
        axes[r, n_cond].imshow(_to_uint8_image(pred_next[r]))
        axes[r, n_cond].set_title("pred", fontsize=8) if r == 0 else None
        axes[r, n_cond].axis("off")
        axes[r, n_cond + 1].imshow(_to_uint8_image(gt_next[r]))
        axes[r, n_cond + 1].set_title("gt", fontsize=8) if r == 0 else None
        axes[r, n_cond + 1].axis("off")
        # diff in [0,1] → uint8 grayscale-ish
        diff_img = (diff[r].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        axes[r, n_cond + 2].imshow(diff_img)
        axes[r, n_cond + 2].set_title("|diff|", fontsize=8) if r == 0 else None
        axes[r, n_cond + 2].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


@torch.no_grad()
def run_validation(
    denoiser: Denoiser,
    val_batch: Batch,
    out_dir: Path,
    step: int,
    num_denoising_steps: int = 10,
    s_cond: float = 0.0,
    max_rows: int = 4,
    save_png: bool = True,
) -> dict:
    """Predict on val_batch, save PNG, return scalar metrics.

    Returns dict with: val_mse, val_psnr, val_image_path.
    """
    device = next(denoiser.parameters()).device
    val_batch = val_batch.to(device)
    was_training = denoiser.training
    denoiser.eval()

    n = denoiser.cfg.inner_model.num_steps_conditioning
    pred = rollout_one_step(
        denoiser,
        val_batch,
        num_denoising_steps=num_denoising_steps,
        s_cond=s_cond,
    )
    gt = val_batch.obs[:, n]

    # Pixel-domain MSE / PSNR (images in [-1, 1]).
    mse = ((pred - gt) ** 2).mean().item()
    # MSE in [-1,1] → PSNR with peak=2: PSNR = 20*log10(2) - 10*log10(mse) = ~6.02 - 10*log10(mse).
    psnr = 20.0 * np.log10(2.0) - 10.0 * np.log10(max(mse, 1e-12))

    image_path = None
    if save_png:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig = make_grid(val_batch, pred, n_cond=n, max_rows=max_rows,
                        title=f"step {step}  val_mse={mse:.4f}  val_psnr={psnr:.2f} dB")
        image_path = out_dir / f"val_step_{step:07d}.png"
        fig.savefig(image_path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    if was_training:
        denoiser.train()

    return {
        "val_mse": mse,
        "val_psnr_db": psnr,
        "val_image_path": str(image_path) if image_path else None,
    }


# ---------------------------------------------------------------------------
# Multi-step autoregressive rollout (pred vs. gt GIF)
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_autoregressive(
    denoiser: Denoiser,
    batch: Batch,
    num_steps: int,
    num_denoising_steps: int = 10,
    sigma_min: float = 2e-3,
    sigma_max: float = 5.0,
    s_cond: float = 0.0,
) -> torch.Tensor:
    """Predict `num_steps` frames autoregressively past the conditioning window.

    Returns a tensor of shape (b, num_steps, c, h, w) in [-1, 1].

    For each new frame we pop the oldest conditioning frame and append our
    prediction — this is the same recipe DIAMOND uses for inference. Action
    conditioning re-uses ground-truth actions from the batch so we measure
    *world model* quality, not policy.
    """
    n = denoiser.cfg.inner_model.num_steps_conditioning
    sampler = DiffusionSampler(
        denoiser,
        DiffusionSamplerConfig(
            num_steps_denoising=num_denoising_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=7,
            order=1,
            s_churn=0,
            s_cond=s_cond,
        ),
    )
    obs = batch.obs.contiguous()                     # (b, T, c, h, w)
    act = batch.act.contiguous()                     # (b, T, A)
    T = obs.size(1)
    assert num_steps <= T - n, f"num_steps={num_steps} exceeds available next frames ({T - n})"

    cond_obs = obs[:, :n].clone()                    # (b, n, c, h, w)
    preds: list[torch.Tensor] = []
    for i in range(num_steps):
        cond_act = act[:, i : n + i].contiguous()
        pred, _ = sampler.sample(cond_obs, cond_act)
        preds.append(pred)
        # Slide window: drop oldest cond frame, append pred.
        cond_obs = torch.cat([cond_obs[:, 1:], pred.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)                  # (b, num_steps, c, h, w)


def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(N, C, H, W) in [-1, 1] -> (N, H, W, C) uint8."""
    return (
        frames.detach().float().cpu().clamp(-1, 1).add(1).div(2).mul(255).byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )


def save_rollout_gif(
    pred: torch.Tensor,
    gt: torch.Tensor,
    out_path: Path,
    fps: int = 10,
    max_rows: int = 4,
) -> Path:
    """Save a side-by-side `pred | gt` rollout GIF.

    pred, gt: (b, T, c, h, w) in [-1, 1]. Stacks the first `max_rows` rows
    vertically; pred on the left, gt on the right.
    """
    b = min(max_rows, pred.size(0))
    pred_u8 = _frames_to_uint8(pred[:b].reshape(-1, *pred.shape[2:])).reshape(b, pred.size(1), *pred.shape[3:5], 3)
    gt_u8 = _frames_to_uint8(gt[:b].reshape(-1, *gt.shape[2:])).reshape(b, gt.size(1), *gt.shape[3:5], 3)
    # Stack: pred|gt horizontally per row, then rows vertically.
    side = np.concatenate([pred_u8, gt_u8], axis=3)              # (b, T, h, 2w, 3)
    canvas = side.transpose(1, 0, 2, 3, 4).reshape(side.shape[1], -1, side.shape[3], 3)
    # canvas: (T, b*h, 2w, 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, list(canvas), fps=fps, loop=0)
    return out_path


@torch.no_grad()
def run_rollout_validation(
    denoiser: Denoiser,
    val_batch: Batch,
    out_dir: Path,
    step: int,
    *,
    num_steps: int = 16,
    num_denoising_steps: int = 10,
    s_cond: float = 0.0,
    max_rows: int = 4,
    fps: int = 10,
) -> dict:
    """Run a multi-step autoregressive rollout, save GIF, return paths + metrics.

    Returns dict with: rollout_gif_path, rollout_mse_per_step (list[float]).
    """
    device = next(denoiser.parameters()).device
    val_batch = val_batch.to(device)
    was_training = denoiser.training
    denoiser.eval()

    n = denoiser.cfg.inner_model.num_steps_conditioning
    available = val_batch.obs.size(1) - n
    steps = min(num_steps, available)

    pred = rollout_autoregressive(
        denoiser,
        val_batch,
        num_steps=steps,
        num_denoising_steps=num_denoising_steps,
        s_cond=s_cond,
    )                                                            # (b, steps, c, h, w)
    gt = val_batch.obs[:, n : n + steps]                          # (b, steps, c, h, w)

    mse_per_step = ((pred - gt) ** 2).mean(dim=(0, 2, 3, 4)).tolist()

    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"rollout_step_{step:07d}.gif"
    save_rollout_gif(pred, gt, gif_path, fps=fps, max_rows=max_rows)

    if was_training:
        denoiser.train()

    return {
        "rollout_gif_path": str(gif_path),
        "rollout_mse_per_step": mse_per_step,
        "rollout_steps": steps,
    }
