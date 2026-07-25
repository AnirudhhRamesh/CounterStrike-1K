"""Paired, deterministic DIAMOND action-sensitivity evaluation on CounterStrike-1K.

The confirmatory comparison evaluates true, cross-round shuffled, and zero
actions on the same checkpoint, held-out windows, and diffusion random draws.
Shuffled donors keep the POV slot fixed while moving the complete action
sequence to a different round.  This preserves temporal coherence and avoids
both self-donors and same-round leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR.parents[1], THIS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .dataset import CSDataset, collate_diamond
from .diamond import Batch, Denoiser
from .train import PRESETS, build_denoiser, load_config_defaults
from .visualize import rollout_autoregressive, rollout_one_step


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def reset_rng(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse: float) -> float:
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(max(mse, 1e-12))


def summarize_valid_rollout(
    mse_per_step: list[float],
    *,
    target_source_frames: list[int],
    alive_end_frame: int,
) -> dict:
    """Mask targets after the POV stops being player-controlled."""

    if len(mse_per_step) != len(target_source_frames):
        raise ValueError("rollout metrics and target frames have different lengths")
    valid = [
        int(source_frame) < int(alive_end_frame)
        for source_frame in target_source_frames
    ]
    valid_values = [
        float(value)
        for value, is_valid in zip(mse_per_step, valid, strict=True)
        if is_valid
    ]
    if not valid_values:
        raise ValueError("rollout has no player-controlled target frames")
    return {
        "mse_per_step": [
            float(value) if is_valid else None
            for value, is_valid in zip(mse_per_step, valid, strict=True)
        ],
        "valid_steps": valid,
        "valid_count": len(valid_values),
        "mean": float(np.mean(valid_values)),
        "last": float(valid_values[-1]),
    }


def replace_actions(batch: Batch, actions: torch.Tensor, info: list[dict]) -> Batch:
    return replace(batch, act=actions, info=info)


def slice_time(batch: Batch, stop: int) -> Batch:
    return Batch(
        obs=batch.obs[:, :stop],
        act=batch.act[:, :stop],
        rew=batch.rew[:, :stop],
        end=batch.end[:, :stop],
        trunc=batch.trunc[:, :stop],
        mask_padding=batch.mask_padding[:, :stop],
        info=batch.info,
        segment_ids=batch.segment_ids,
    )


@torch.no_grad()
def denoising_loss_per_sample(denoiser: Denoiser, batch: Batch) -> torch.Tensor:
    """DIAMOND's training objective with reduction retained per sample."""

    b, t, c, h, w = batch.obs.size()
    if denoiser.is_upsampler:
        raise ValueError(
            "CounterStrike-1K action sensitivity expects the base denoiser"
        )
    n = denoiser.cfg.inner_model.num_steps_conditioning
    seq_length = t - n
    if seq_length < 1:
        raise ValueError(f"batch needs at least {n + 1} frames, got {t}")

    all_obs = batch.obs.clone()
    per_sample = torch.zeros(b, device=batch.obs.device, dtype=torch.float32)
    for i in range(seq_length):
        prev_obs = all_obs[:, i : n + i].reshape(b, n * c, h, w)
        prev_act = batch.act[:, i : n + i]
        obs = all_obs[:, n + i]
        mask = batch.mask_padding[:, n + i]

        if denoiser.cfg.noise_previous_obs:
            sigma_cond = denoiser.sample_sigma_training(b, denoiser.device)
            prev_obs = denoiser.apply_noise(
                prev_obs,
                sigma_cond,
                denoiser.cfg.sigma_offset_noise,
            )
        else:
            sigma_cond = None
        sigma = denoiser.sample_sigma_training(b, denoiser.device)
        noisy_obs = denoiser.apply_noise(obs, sigma, denoiser.cfg.sigma_offset_noise)
        conditioners = denoiser.compute_conditioners(sigma, sigma_cond)
        model_output = denoiser.compute_model_output(
            noisy_obs,
            prev_obs,
            prev_act,
            conditioners,
        )
        target = (obs - conditioners.c_skip * noisy_obs) / conditioners.c_out
        step_loss = F.mse_loss(model_output, target, reduction="none").mean((1, 2, 3))
        per_sample += torch.where(mask, step_loss, torch.zeros_like(step_loss))
        all_obs[:, n + i] = denoiser.wrap_model_output(
            noisy_obs,
            model_output,
            conditioners,
        )
    return per_sample / seq_length


def build_cross_round_donors(infos: list[dict], seed: int) -> list[int]:
    """Map every row to the same POV in a different held-out round."""

    by_round_pov: dict[tuple[str, int], int] = {}
    round_ids: set[str] = set()
    for idx, info in enumerate(infos):
        round_id = str(info["round_id"])
        pov_idx = int(info["pov_idx"])
        key = (round_id, pov_idx)
        if not round_id or key in by_round_pov:
            raise ValueError(f"invalid or duplicate held-out round/POV key: {key}")
        by_round_pov[key] = idx
        round_ids.add(round_id)

    rounds = sorted(round_ids)
    if len(rounds) < 2:
        raise ValueError("cross-round shuffle needs at least two held-out rounds")
    pov_sets = {
        round_id: {
            pov_idx
            for candidate_round, pov_idx in by_round_pov
            if candidate_round == round_id
        }
        for round_id in rounds
    }
    expected_povs = next(iter(pov_sets.values()))
    if any(povs != expected_povs for povs in pov_sets.values()):
        raise ValueError(
            f"held-out rounds do not share one complete POV set: {pov_sets}"
        )

    generator = torch.Generator().manual_seed(seed)
    shift = int(torch.randint(1, len(rounds), (1,), generator=generator).item())
    donor_round = {
        round_id: rounds[(idx + shift) % len(rounds)]
        for idx, round_id in enumerate(rounds)
    }
    donors = [
        by_round_pov[(donor_round[str(info["round_id"])], int(info["pov_idx"]))]
        for info in infos
    ]
    if any(
        i == donor or infos[i]["round_id"] == infos[donor]["round_id"]
        for i, donor in enumerate(donors)
    ):
        raise AssertionError(
            "cross-round donor mapping contains a fixed or same-round point"
        )
    return donors


def select_dataset_indices(
    dataset: CSDataset,
    *,
    pov_idx: int | None,
    max_samples: int | None,
) -> list[int]:
    """Select deterministic evaluator rows without changing dataset ordering."""

    indices = list(range(len(dataset)))
    if pov_idx is not None:
        indices = [
            index
            for index in indices
            if int(dataset._resolve_window(index)[0].get("pov_idx", -1)) == pov_idx
        ]
        if not indices:
            raise ValueError(f"no evaluator rows have pov_idx={pov_idx}")
    if max_samples is not None:
        indices = indices[:max_samples]
    return indices


def _checkpoint_args(
    checkpoint: dict,
    cli_args: argparse.Namespace,
    config_payload: dict | None,
) -> SimpleNamespace:
    defaults = {}
    if config_payload is not None:
        defaults.update(config_payload.get("train_args", {}))
    defaults.update(checkpoint.get("args", {}))
    defaults["data_dir"] = str(cli_args.data_dir)
    defaults["out_dir"] = str(cli_args.out_dir)
    defaults["device"] = cli_args.device
    return SimpleNamespace(**defaults)


def _uint8_video(frames: torch.Tensor) -> np.ndarray:
    return (
        frames.detach()
        .float()
        .cpu()
        .clamp(-1, 1)
        .add(1)
        .div(2)
        .mul(255)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )


def save_review_video(
    *,
    out_path: Path,
    ground_truth: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    fps: int,
) -> None:
    columns = [_uint8_video(ground_truth)]
    columns.extend(_uint8_video(predictions[mode]) for mode in predictions)
    canvas = np.concatenate(columns, axis=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        out_path,
        list(canvas),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def paired_summary(
    rows: list[dict],
    *,
    metric: str,
    reference: str = "true",
    bootstrap_seed: int = 20250725,
    n_bootstrap: int = 10_000,
) -> dict:
    by_key: dict[tuple[int, int, str], dict[str, float]] = defaultdict(dict)
    round_for_key: dict[tuple[int, int, str], str] = {}
    for row in rows:
        key = (int(row["eval_seed"]), int(row["dataset_index"]), str(row["sample_key"]))
        by_key[key][str(row["action_mode"])] = float(row[metric])
        round_for_key[key] = str(row["round_id"])

    output = {}
    for mode in sorted({str(row["action_mode"]) for row in rows} - {reference}):
        deltas_by_round: dict[str, list[float]] = defaultdict(list)
        for key, values in by_key.items():
            if reference in values and mode in values:
                # Positive MSE delta means true actions are better.
                deltas_by_round[round_for_key[key]].append(
                    values[mode] - values[reference]
                )
        round_means = np.asarray(
            [np.mean(values) for values in deltas_by_round.values()],
            dtype=np.float64,
        )
        if round_means.size == 0:
            raise ValueError(f"no paired {reference}/{mode} rows for {metric}")
        rng = np.random.default_rng(bootstrap_seed)
        indices = rng.integers(
            0,
            round_means.size,
            size=(n_bootstrap, round_means.size),
        )
        bootstrap = round_means[indices].mean(axis=1)
        output[mode] = {
            "delta_definition": f"{mode}_minus_{reference}",
            "mean_delta": float(round_means.mean()),
            "cluster_unit": "round_id",
            "num_round_clusters": int(round_means.size),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": n_bootstrap,
            "ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "fraction_rounds_positive": float((round_means > 0).mean()),
        }
    return output


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--manifest-name",
        default="manifest_dust2_confirmatory_spatial_v1.parquet",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--map-slug", default="dust2")
    parser.add_argument(
        "--window-mode",
        choices=["midpoint", "first-death"],
        default="midpoint",
    )
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[37, 41, 43])
    parser.add_argument(
        "--action-modes",
        nargs="+",
        choices=["true", "shuffled", "zeros"],
        default=["true", "shuffled", "zeros"],
    )
    parser.add_argument("--num-denoising-steps", type=int)
    parser.add_argument("--s-cond", type=float)
    parser.add_argument("--weights", choices=["raw", "ema", "auto"], default="auto")
    parser.add_argument("--expected-samples", type=int, default=690)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--pov-idx",
        type=int,
        default=None,
        help=(
            "Optional fixed POV slot for validation-only convergence audits. "
            "The confirmatory test protocol leaves this unset."
        ),
    )
    parser.add_argument("--max-review-videos", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    config_payload = None
    config_defaults = {}
    if args.config:
        config_payload, config_defaults = load_config_defaults(args.config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = _checkpoint_args(checkpoint, args, config_payload)
    preset = dict(PRESETS[model_args.preset])
    if getattr(model_args, "resize", None) is not None:
        preset["resize"] = tuple(model_args.resize)
    model = build_denoiser(model_args, preset, device)

    weights = args.weights
    if weights == "auto":
        weights = "ema" if "ema" in checkpoint else "raw"
    if weights == "ema":
        if "ema" not in checkpoint:
            raise ValueError(f"{args.checkpoint} has no EMA weights")
        model.load_state_dict(checkpoint["ema"]["shadow"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    denoising_steps = (
        args.num_denoising_steps
        if args.num_denoising_steps is not None
        else int(
            config_defaults.get(
                "val_denoise_steps",
                getattr(model_args, "val_denoise_steps", 1),
            )
        )
    )
    s_cond = (
        args.s_cond
        if args.s_cond is not None
        else float(
            config_defaults.get("val_s_cond", getattr(model_args, "val_s_cond", 0.005))
        )
    )
    n_cond = model.cfg.inner_model.num_steps_conditioning
    dataset = CSDataset(
        data_path=args.data_dir,
        split=args.split,
        T=n_cond + args.rollout_steps,
        stride=getattr(model_args, "stride", None),
        target_fps=int(getattr(model_args, "target_fps", 8)),
        resize=tuple(preset["resize"]),
        manifest_name=args.manifest_name,
        mode="diamond",
        window_mode=args.window_mode,
        map_slug=args.map_slug,
        resolution=getattr(model_args, "resolution", "360p"),
        verify_sha256=bool(getattr(model_args, "verify_sha256", False)),
    )
    selected_indices = select_dataset_indices(
        dataset,
        pov_idx=args.pov_idx,
        max_samples=args.max_samples,
    )
    if args.max_samples is None and len(selected_indices) != args.expected_samples:
        raise ValueError(
            f"confirmatory evaluation expected {args.expected_samples} held-out POV rows, "
            f"found {len(selected_indices)}"
        )
    num_samples = len(selected_indices)
    infos: list[dict] = []
    action_windows: list[torch.Tensor] = []
    for dataset_index in selected_indices:
        actions, info = dataset.action_window_at(dataset_index)
        action_windows.append(actions)
        infos.append(info)
    donors = build_cross_round_donors(infos, seed=args.eval_seeds[0] + 90_001)
    donor_actions = [action_windows[donor] for donor in donors]
    position_by_dataset_index = {
        int(info["dataset_index"]): position for position, info in enumerate(infos)
    }

    sample_plan = [
        {
            **info,
            "action_donor_dataset_index": int(
                infos[donors[idx]]["dataset_index"]
            ),
            "action_donor_sample_key": infos[donors[idx]]["sample_key"],
            "action_donor_round_id": infos[donors[idx]]["round_id"],
        }
        for idx, info in enumerate(infos)
    ]
    sample_plan_path = args.out_dir / "sample_plan.json"
    sample_plan_path.write_text(json.dumps(sample_plan, indent=2), encoding="utf-8")

    loader = DataLoader(
        torch.utils.data.Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_diamond,
    )

    rows: list[dict] = []
    review_manifest: list[dict] = []
    review_count = 0
    for batch_index, cpu_batch in enumerate(loader):
        batch = cpu_batch.to(device)
        dataset_indices = [int(info["dataset_index"]) for info in batch.info]
        positions = [
            position_by_dataset_index[dataset_index]
            for dataset_index in dataset_indices
        ]
        shuffled_actions = torch.stack(
            [donor_actions[position] for position in positions],
        ).to(device)
        modes: dict[str, Batch] = {
            "true": batch,
            "shuffled": replace_actions(
                batch,
                shuffled_actions,
                [
                    {
                        **info,
                        "action_source_dataset_index": int(
                            infos[donors[position]]["dataset_index"]
                        ),
                        "action_source_sample_key": infos[donors[position]][
                            "sample_key"
                        ],
                        "action_source_round_id": infos[donors[position]]["round_id"],
                    }
                    for info, position in zip(batch.info, positions, strict=True)
                ],
            ),
            "zeros": replace_actions(
                batch,
                torch.zeros_like(batch.act),
                [{**info, "action_source_dataset_index": None} for info in batch.info],
            ),
        }
        predictions_for_review: dict[str, torch.Tensor] = {}
        for eval_seed in args.eval_seeds:
            for mode in args.action_modes:
                mode_batch = modes[mode]
                rng_base = eval_seed * 1_000_003 + batch_index * 101
                reset_rng(rng_base + 1, device)
                denoising = denoising_loss_per_sample(
                    model,
                    slice_time(mode_batch, n_cond + 1),
                )
                reset_rng(rng_base + 2, device)
                one_step = rollout_one_step(
                    model,
                    mode_batch,
                    num_denoising_steps=denoising_steps,
                    s_cond=s_cond,
                )
                one_step_mse = ((one_step - mode_batch.obs[:, n_cond]) ** 2).mean(
                    (1, 2, 3)
                )
                reset_rng(rng_base + 3, device)
                rollout = rollout_autoregressive(
                    model,
                    mode_batch,
                    num_steps=args.rollout_steps,
                    num_denoising_steps=denoising_steps,
                    s_cond=s_cond,
                )
                ground_truth = mode_batch.obs[:, n_cond : n_cond + args.rollout_steps]
                rollout_mse_steps = ((rollout - ground_truth) ** 2).mean((2, 3, 4))

                if eval_seed == args.eval_seeds[0]:
                    predictions_for_review[mode] = rollout.detach().cpu()
                for row_idx, (dataset_index, position) in enumerate(
                    zip(dataset_indices, positions, strict=True)
                ):
                    info = infos[position]
                    rollout_values_raw = (
                        rollout_mse_steps[row_idx].detach().float().cpu().tolist()
                    )
                    target_source_frames = info["source_frame_indices"][
                        n_cond : n_cond + args.rollout_steps
                    ]
                    rollout_metrics = summarize_valid_rollout(
                        rollout_values_raw,
                        target_source_frames=target_source_frames,
                        alive_end_frame=int(info["alive_end_frame"]),
                    )
                    if int(target_source_frames[0]) >= int(info["alive_end_frame"]):
                        raise ValueError(
                            f"{info['sample_key']}: one-step target is post-alive"
                        )
                    rows.append(
                        {
                            "eval_seed": eval_seed,
                            "batch_index": batch_index,
                            "dataset_index": dataset_index,
                            "sample_key": info["sample_key"],
                            "match_id": info["match_id"],
                            "round_id": info["round_id"],
                            "pov_idx": info["pov_idx"],
                            "window_mode": args.window_mode,
                            "action_mode": mode,
                            "action_source_sample_key": (
                                info["sample_key"]
                                if mode == "true"
                                else infos[donors[position]]["sample_key"]
                                if mode == "shuffled"
                                else None
                            ),
                            "denoising_loss": float(denoising[row_idx].item()),
                            "one_step_mse": float(one_step_mse[row_idx].item()),
                            "one_step_psnr_db": psnr_from_mse(
                                float(one_step_mse[row_idx].item())
                            ),
                            "rollout_target_source_frames": target_source_frames,
                            "rollout_valid_steps": rollout_metrics["valid_steps"],
                            "rollout_valid_count": rollout_metrics["valid_count"],
                            "rollout_mse_per_step": rollout_metrics["mse_per_step"],
                            "rollout_mse_mean": rollout_metrics["mean"],
                            "rollout_mse_last": rollout_metrics["last"],
                        }
                    )

        if (
            args.eval_seeds
            and all(mode in predictions_for_review for mode in args.action_modes)
            and review_count < args.max_review_videos
        ):
            ground_truth = batch.obs[:, n_cond : n_cond + args.rollout_steps].cpu()
            for row_idx, (dataset_index, position) in enumerate(
                zip(dataset_indices, positions, strict=True)
            ):
                if review_count >= args.max_review_videos:
                    break
                safe_key = str(infos[position]["sample_key"]).replace("/", "_")
                relative_path = Path("review") / f"{review_count:04d}_{safe_key}.mp4"
                save_review_video(
                    out_path=args.out_dir / relative_path,
                    ground_truth=ground_truth[row_idx],
                    predictions={
                        mode: predictions_for_review[mode][row_idx]
                        for mode in args.action_modes
                    },
                    fps=int(getattr(model_args, "target_fps", 8)),
                )
                review_manifest.append(
                    {
                        "path": relative_path.as_posix(),
                        "dataset_index": dataset_index,
                        "sample_key": infos[position]["sample_key"],
                        "round_id": infos[position]["round_id"],
                        "pov_idx": infos[position]["pov_idx"],
                        "eval_seed": args.eval_seeds[0],
                        "columns": ["ground_truth", *args.action_modes],
                    }
                )
                review_count += 1

    rows_path = args.out_dir / "per_sample_metrics.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.out_dir / "review_manifest.json").write_text(
        json.dumps(review_manifest, indent=2),
        encoding="utf-8",
    )

    means = {}
    for mode in args.action_modes:
        mode_rows = [row for row in rows if row["action_mode"] == mode]
        means[mode] = {
            metric: float(np.mean([row[metric] for row in mode_rows]))
            for metric in (
                "denoising_loss",
                "one_step_mse",
                "one_step_psnr_db",
                "rollout_mse_mean",
                "rollout_mse_last",
            )
        }
    summary = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "weights": weights,
        "config": str(args.config) if args.config else None,
        "config_sha256": sha256_file(args.config) if args.config else None,
        "manifest": str(args.data_dir / args.manifest_name),
        "manifest_sha256": sha256_file(args.data_dir / args.manifest_name),
        "split": args.split,
        "map_slug": args.map_slug,
        "window_mode": args.window_mode,
        "target_fps": int(getattr(model_args, "target_fps", 8)),
        "pov_idx_filter": args.pov_idx,
        "resize": list(preset["resize"]),
        "rollout_steps": args.rollout_steps,
        "rollout_target_masking": "source_frame < alive_end_frame",
        "num_denoising_steps": denoising_steps,
        "s_cond": s_cond,
        "eval_seeds": args.eval_seeds,
        "action_modes": args.action_modes,
        "num_eval_samples": num_samples,
        "num_rounds": len({info["round_id"] for info in infos}),
        "sample_plan": str(sample_plan_path),
        "sample_plan_sha256": sha256_file(sample_plan_path),
        "action_plan_sha256": sha256_json(
            [
                {
                    "target": info["sample_key"],
                    "donor": infos[donors[idx]]["sample_key"],
                }
                for idx, info in enumerate(infos)
            ]
        ),
        "means": means,
        "paired_deltas": {
            metric: paired_summary(rows, metric=metric)
            for metric in (
                "denoising_loss",
                "one_step_mse",
                "rollout_mse_mean",
                "rollout_mse_last",
            )
        },
        "review_manifest": str(args.out_dir / "review_manifest.json"),
        "per_sample_metrics": str(rows_path),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
