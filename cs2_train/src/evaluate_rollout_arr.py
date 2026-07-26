"""Evaluate archived rollouts with temporal CS2 Action Recoverability Ratio.

The frozen probe predicts actions from two 0.5-second video segments. ARR is
per-class generated-video AP divided by the model-native ground-truth-video AP
ceiling. The scorer reports adherence to each rollout's own conditioning
stream, alignment with the true stream, and shuffled-action redirection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_rollout_motion import verify_archive
from .rollout_archive import sha256_file
from .temporal_action_probe import (
    ACTION_LABEL_NAMES,
    MOUSE_DIRECTION_THRESHOLD_DEGREES,
    DinoV2FrameEncoder,
    TemporalActionProbe,
    actions_to_labels_numpy,
    binary_average_precision_tie_aware,
)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def nanmean_rows(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    return np.divide(
        np.where(finite, values, 0).sum(axis=1),
        counts,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def segment_labels(actions: np.ndarray) -> np.ndarray:
    """Return ``[mode,sample,segment,class]`` labels from ``[...,8,14]``."""

    if actions.ndim != 4 or actions.shape[-2:] != (8, 14):
        raise ValueError(f"expected [mode,sample,8,14], got {actions.shape}")
    return np.stack(
        (
            actions_to_labels_numpy(actions[:, :, 0:4]),
            actions_to_labels_numpy(actions[:, :, 4:8]),
        ),
        axis=2,
    ).astype(np.float32)


def segment_validity(valid_steps: np.ndarray) -> np.ndarray:
    if valid_steps.ndim != 2 or valid_steps.shape[1] != 8:
        raise ValueError(f"expected [sample,8] validity, got {valid_steps.shape}")
    return np.stack(
        (
            valid_steps[:, 0:4].all(axis=1),
            valid_steps[:, 4:8].all(axis=1),
        ),
        axis=1,
    )


def weighted_cluster_bootstrap_ap(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_indices: np.ndarray,
    cluster_counts: np.ndarray,
    *,
    replicate_batch_size: int = 256,
) -> np.ndarray:
    """Compute AP for cluster-bootstrap multiplicities without row expansion."""

    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    cluster_indices = np.asarray(cluster_indices, dtype=np.int64)
    cluster_counts = np.asarray(cluster_counts)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be equal-length vectors")
    if cluster_indices.shape != labels.shape:
        raise ValueError("cluster indices must match labels")
    if cluster_counts.ndim != 2:
        raise ValueError("cluster counts must be [replicate,cluster]")
    if len(labels) == 0:
        raise ValueError("cannot bootstrap an empty prediction set")
    if cluster_indices.min() < 0 or cluster_indices.max() >= cluster_counts.shape[1]:
        raise ValueError("cluster index is outside the count matrix")
    if replicate_batch_size <= 0:
        raise ValueError("replicate batch size must be positive")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_clusters = cluster_indices[order]
    group_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    output = np.full(len(cluster_counts), np.nan, dtype=np.float64)
    for start in range(0, len(cluster_counts), replicate_batch_size):
        stop = min(start + replicate_batch_size, len(cluster_counts))
        weights = cluster_counts[start:stop, sorted_clusters].astype(
            np.float64,
            copy=False,
        )
        positive_weights = weights * sorted_labels[None]
        total_positive = positive_weights.sum(axis=1)
        cumulative_positive = np.cumsum(positive_weights, axis=1)
        cumulative_weight = np.cumsum(weights, axis=1)
        positive_at_threshold = cumulative_positive[:, group_ends]
        weight_at_threshold = cumulative_weight[:, group_ends]
        group_positive = np.diff(
            np.concatenate(
                (
                    np.zeros((stop - start, 1), dtype=np.float64),
                    positive_at_threshold,
                ),
                axis=1,
            ),
            axis=1,
        )
        precision = np.divide(
            positive_at_threshold,
            weight_at_threshold,
            out=np.zeros_like(positive_at_threshold),
            where=weight_at_threshold > 0,
        )
        numerator = (precision * group_positive).sum(axis=1)
        output[start:stop] = np.divide(
            numerator,
            total_positive,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=total_positive > 0,
        )
    return output


def class_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("labels/scores must be equal [example,class] arrays")
    return np.asarray(
        [
            binary_average_precision_tie_aware(
                labels[:, index],
                scores[:, index],
            )
            for index in range(labels.shape[1])
        ],
        dtype=np.float64,
    )


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.isfinite(denominator) & (denominator > 0),
    )


def load_probe(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[TemporalActionProbe, dict]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported temporal action-probe checkpoint")
    if checkpoint["label_names"] != list(ACTION_LABEL_NAMES):
        raise ValueError("temporal action-probe label contract differs")
    if (
        checkpoint["label_contract"]["mouse_threshold_degrees"]
        != MOUSE_DIRECTION_THRESHOLD_DEGREES
    ):
        raise ValueError("temporal action-probe mouse threshold differs")
    model = TemporalActionProbe(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def score_video_batch(
    video_uint8: np.ndarray,
    *,
    encoder: DinoV2FrameEncoder,
    probe: TemporalActionProbe,
    device: torch.device,
) -> np.ndarray:
    """Return ``[sample,2,class]`` scores for ``[sample,9,H,W,3]`` video."""

    video = torch.from_numpy(np.array(video_uint8, copy=True)).permute(0, 1, 4, 2, 3)
    features = encoder.encode_video(video)
    segments = torch.stack((features[:, 0:5], features[:, 4:9]), dim=1)
    logits = probe(segments.reshape(-1, 5, features.shape[-1]).to(device))
    return (
        torch.sigmoid(logits)
        .reshape(len(video), 2, len(ACTION_LABEL_NAMES))
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def summarize_arr(
    *,
    real_scores: np.ndarray,
    generated_scores: np.ndarray,
    labels_by_mode: np.ndarray,
    valid_segments: np.ndarray,
    action_modes: list[str],
    round_ids: list[str],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict:
    """Compute point estimates and paired round-bootstrap confidence intervals."""

    num_classes = len(ACTION_LABEL_NAMES)
    if real_scores.ndim != 3 or real_scores.shape[1:] != (2, num_classes):
        raise ValueError(
            f"real scores must be [sample,2,{num_classes}], got {real_scores.shape}"
        )
    num_samples = real_scores.shape[0]
    if (
        generated_scores.ndim != 5
        or generated_scores.shape[0] <= 0
        or generated_scores.shape[1:]
        != (
            len(action_modes),
            num_samples,
            2,
            num_classes,
        )
    ):
        raise ValueError("generated-score axes do not match the ARR contract")
    if labels_by_mode.shape != (
        len(action_modes),
        num_samples,
        2,
        num_classes,
    ):
        raise ValueError("action-label axes do not match the ARR contract")
    if valid_segments.shape != (num_samples, 2):
        raise ValueError("valid-segment axes do not match the ARR contract")
    if valid_segments.dtype != np.bool_:
        raise ValueError("valid segments must be a boolean array")
    if len(round_ids) != num_samples or any(not value for value in round_ids):
        raise ValueError("every ARR sample needs a non-empty round id")
    if len(set(action_modes)) != len(action_modes):
        raise ValueError("ARR action modes must be unique")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    for name, scores in (
        ("real", real_scores),
        ("generated", generated_scores),
    ):
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"{name} probe scores must be finite probabilities")
    if (
        not np.isfinite(labels_by_mode).all()
        or not np.isin(labels_by_mode, (0, 1)).all()
    ):
        raise ValueError("ARR action labels must be finite and binary")
    if "true" not in action_modes or "shuffled" not in action_modes:
        raise ValueError("ARR requires true and shuffled action modes")
    true_index = action_modes.index("true")
    valid_flat = valid_segments.reshape(-1)
    if not valid_flat.any():
        raise ValueError("no complete action-probe segments are valid")
    true_labels = labels_by_mode[true_index].reshape(-1, len(ACTION_LABEL_NAMES))[
        valid_flat
    ]
    real = real_scores.reshape(-1, len(ACTION_LABEL_NAMES))[valid_flat]
    real_ap = class_average_precision(true_labels, real)

    cluster_names = sorted(set(round_ids))
    if len(cluster_names) < 2:
        raise ValueError("ARR confidence intervals require at least two rounds")
    cluster_lookup = {name: index for index, name in enumerate(cluster_names)}
    real_cluster_indices = np.repeat(
        np.asarray([cluster_lookup[name] for name in round_ids], dtype=np.int64),
        2,
    )[valid_flat]
    rng = np.random.default_rng(bootstrap_seed)
    cluster_counts = rng.multinomial(
        len(cluster_names),
        np.full(len(cluster_names), 1 / len(cluster_names)),
        size=bootstrap_replicates,
    )
    real_ap_bootstrap = np.stack(
        [
            weighted_cluster_bootstrap_ap(
                true_labels[:, class_index],
                real[:, class_index],
                real_cluster_indices,
                cluster_counts,
            )
            for class_index in range(len(ACTION_LABEL_NAMES))
        ],
        axis=1,
    )

    mode_results = {}
    bootstrap_mode = {}
    eval_seeds = generated_scores.shape[0]
    for mode_index, mode in enumerate(action_modes):
        generated = (
            generated_scores[:, mode_index]
            .reshape(
                eval_seeds,
                -1,
                len(ACTION_LABEL_NAMES),
            )[:, valid_flat]
            .reshape(-1, len(ACTION_LABEL_NAMES))
        )
        own_labels_base = labels_by_mode[mode_index].reshape(
            -1, len(ACTION_LABEL_NAMES)
        )[valid_flat]
        own_labels = np.tile(own_labels_base, (eval_seeds, 1))
        target_labels = np.tile(true_labels, (eval_seeds, 1))
        generated_clusters = np.tile(real_cluster_indices, eval_seeds)
        own_ap = class_average_precision(own_labels, generated)
        true_target_ap = class_average_precision(target_labels, generated)
        own_arr = ratio(own_ap, real_ap)
        true_target_arr = ratio(true_target_ap, real_ap)

        per_class = {}
        for class_index, name in enumerate(ACTION_LABEL_NAMES):
            per_class[name] = {
                "real_ceiling_average_precision": json_number(real_ap[class_index]),
                "own_conditioning_average_precision": json_number(own_ap[class_index]),
                "own_conditioning_arr": json_number(own_arr[class_index]),
                "true_target_average_precision": json_number(
                    true_target_ap[class_index]
                ),
                "true_target_arr": json_number(true_target_arr[class_index]),
                "own_positive_examples": int(own_labels[:, class_index].sum()),
                "true_positive_examples": int(target_labels[:, class_index].sum()),
            }
        mode_results[mode] = {
            "macro_real_ceiling_average_precision": json_number(
                finite_mean(real_ap.tolist())
            ),
            "macro_own_conditioning_average_precision": json_number(
                finite_mean(own_ap.tolist())
            ),
            "macro_own_conditioning_arr": json_number(finite_mean(own_arr.tolist())),
            "macro_true_target_average_precision": json_number(
                finite_mean(true_target_ap.tolist())
            ),
            "macro_true_target_arr": json_number(finite_mean(true_target_arr.tolist())),
            "per_class": per_class,
        }

        own_ap_bootstrap = np.stack(
            [
                weighted_cluster_bootstrap_ap(
                    own_labels[:, class_index],
                    generated[:, class_index],
                    generated_clusters,
                    cluster_counts,
                )
                for class_index in range(len(ACTION_LABEL_NAMES))
            ],
            axis=1,
        )
        true_ap_bootstrap = np.stack(
            [
                weighted_cluster_bootstrap_ap(
                    target_labels[:, class_index],
                    generated[:, class_index],
                    generated_clusters,
                    cluster_counts,
                )
                for class_index in range(len(ACTION_LABEL_NAMES))
            ],
            axis=1,
        )
        bootstrap_mode[mode] = {
            "own": nanmean_rows(ratio(own_ap_bootstrap, real_ap_bootstrap)),
            "true_target": nanmean_rows(ratio(true_ap_bootstrap, real_ap_bootstrap)),
        }

    primary_draws = {
        "true_own_conditioning_arr": bootstrap_mode["true"]["own"],
        "shuffled_own_conditioning_arr": bootstrap_mode["shuffled"]["own"],
        "true_target_alignment_separation": (
            bootstrap_mode["true"]["true_target"]
            - bootstrap_mode["shuffled"]["true_target"]
        ),
        "shuffled_action_redirection": (
            bootstrap_mode["shuffled"]["own"]
            - bootstrap_mode["shuffled"]["true_target"]
        ),
    }
    if "zeros" in bootstrap_mode:
        primary_draws["true_vs_zero_target_alignment_separation"] = (
            bootstrap_mode["true"]["true_target"]
            - bootstrap_mode["zeros"]["true_target"]
        )

    primary = {}
    for name, draws in primary_draws.items():
        finite = draws[np.isfinite(draws)]
        if not len(finite):
            primary[name] = {
                "estimate": None,
                "ci95": [None, None],
                "finite_bootstrap_replicates": 0,
            }
            continue
        if name == "true_own_conditioning_arr":
            estimate = mode_results["true"]["macro_own_conditioning_arr"]
        elif name == "shuffled_own_conditioning_arr":
            estimate = mode_results["shuffled"]["macro_own_conditioning_arr"]
        elif name == "true_target_alignment_separation":
            estimate = (
                mode_results["true"]["macro_true_target_arr"]
                - mode_results["shuffled"]["macro_true_target_arr"]
            )
        elif name == "shuffled_action_redirection":
            estimate = (
                mode_results["shuffled"]["macro_own_conditioning_arr"]
                - mode_results["shuffled"]["macro_true_target_arr"]
            )
        else:
            estimate = (
                mode_results["true"]["macro_true_target_arr"]
                - mode_results["zeros"]["macro_true_target_arr"]
            )
        primary[name] = {
            "estimate": estimate,
            "ci95": [
                float(np.quantile(finite, 0.025)),
                float(np.quantile(finite, 0.975)),
            ],
            "finite_bootstrap_replicates": len(finite),
        }
    return {
        "real_ceiling_per_class_average_precision": {
            name: json_number(real_ap[index])
            for index, name in enumerate(ACTION_LABEL_NAMES)
        },
        "modes": mode_results,
        "primary": primary,
        "bootstrap": {
            "unit": "round_id",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "round_clusters": len(cluster_names),
            "same_cluster_draws_for_real_and_all_action_modes": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-seed", type=int, default=20250728)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.sample_batch_size <= 0 or args.frame_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    out_dir = args.out_dir or args.archive_dir.parent / "action_recoverability"
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"ARR output directory is not empty: {out_dir}")

    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    archive_metadata = verify_archive(args.archive_dir)
    artifacts = archive_metadata["artifacts"]
    predictions = np.load(
        args.archive_dir / artifacts["predictions_uint8"]["path"],
        mmap_mode="r",
    )
    ground_truth = np.load(
        args.archive_dir / artifacts["ground_truth_uint8"]["path"],
        mmap_mode="r",
    )
    context_last = np.load(
        args.archive_dir / artifacts["context_last_uint8"]["path"],
        mmap_mode="r",
    )
    actions = np.load(
        args.archive_dir / artifacts["conditioning_actions_cs2_float32"]["path"],
        mmap_mode="r",
    )
    valid_steps = np.load(
        args.archive_dir / artifacts["valid_steps_bool"]["path"],
        mmap_mode="r",
    )
    sample_plan_path = args.archive_dir.parent / "sample_plan.json"
    if (
        sha256_file(sample_plan_path)
        != archive_metadata["contract"]["sample_plan_sha256"]
    ):
        raise ValueError("rollout archive sample-plan hash mismatch")
    sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))
    contract = archive_metadata["contract"]
    eval_seeds = contract["eval_seeds"]
    action_modes = contract["action_modes"]
    if (
        contract["rollout_steps"] != 8
        or contract["num_cs2_actions"] != 14
        or len(eval_seeds) <= 0
        or len(set(eval_seeds)) != len(eval_seeds)
        or len(set(action_modes)) != len(action_modes)
    ):
        raise ValueError("rollout archive is incompatible with ARR v1")
    expected_num_samples = int(contract["num_samples"])
    if len(sample_plan) != expected_num_samples:
        raise ValueError("sample plan length differs from rollout archive")
    expected_prediction_shape = (
        len(eval_seeds),
        len(action_modes),
        expected_num_samples,
        8,
        int(contract["height"]),
        int(contract["width"]),
        3,
    )
    if predictions.shape != expected_prediction_shape:
        raise ValueError("prediction array axes differ from rollout contract")
    if ground_truth.shape != expected_prediction_shape[2:]:
        raise ValueError("ground-truth array axes differ from rollout contract")
    if context_last.shape != (
        expected_num_samples,
        int(contract["height"]),
        int(contract["width"]),
        3,
    ):
        raise ValueError("context-frame array axes differ from rollout contract")
    if actions.shape != (
        len(action_modes),
        expected_num_samples,
        8,
        14,
    ):
        raise ValueError("CS2-action array axes differ from rollout contract")
    if valid_steps.shape != (expected_num_samples, 8):
        raise ValueError("valid-step array axes differ from rollout contract")
    num_samples = len(sample_plan)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    if predictions.shape[2] < num_samples:
        raise ValueError("rollout archive has fewer rows than the sample plan")

    device = torch.device(args.device)
    probe, probe_checkpoint = load_probe(
        args.probe_checkpoint,
        device=device,
    )
    encoder = DinoV2FrameEncoder(
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    expected_backbone = probe_checkpoint["backbone"]
    actual_backbone = encoder.provenance()
    for key in ("repository", "model", "weights_sha256", "preprocess"):
        if expected_backbone[key] != actual_backbone[key]:
            raise ValueError(f"probe/evaluator backbone drift at {key}")

    num_classes = len(ACTION_LABEL_NAMES)
    real_scores = np.empty((num_samples, 2, num_classes), dtype=np.float32)
    generated_scores = np.empty(
        (
            len(eval_seeds),
            len(action_modes),
            num_samples,
            2,
            num_classes,
        ),
        dtype=np.float32,
    )
    for start in range(0, num_samples, args.sample_batch_size):
        stop = min(start + args.sample_batch_size, num_samples)
        real_video = np.concatenate(
            [context_last[start:stop, None], ground_truth[start:stop]],
            axis=1,
        )
        real_scores[start:stop] = score_video_batch(
            real_video,
            encoder=encoder,
            probe=probe,
            device=device,
        )
        for seed_index in range(len(eval_seeds)):
            for mode_index in range(len(action_modes)):
                generated_video = np.concatenate(
                    [
                        context_last[start:stop, None],
                        predictions[
                            seed_index,
                            mode_index,
                            start:stop,
                        ],
                    ],
                    axis=1,
                )
                generated_scores[
                    seed_index,
                    mode_index,
                    start:stop,
                ] = score_video_batch(
                    generated_video,
                    encoder=encoder,
                    probe=probe,
                    device=device,
                )

    labels_by_mode = segment_labels(actions[:, :num_samples])
    valid_segments = segment_validity(valid_steps[:num_samples])
    arr = summarize_arr(
        real_scores=real_scores,
        generated_scores=generated_scores,
        labels_by_mode=labels_by_mode,
        valid_segments=valid_segments,
        action_modes=action_modes,
        round_ids=[str(sample_plan[index]["round_id"]) for index in range(num_samples)],
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )

    arrays = {
        "real_scores_float32": real_scores,
        "generated_scores_float32": generated_scores,
        "labels_by_mode_float32": labels_by_mode,
        "valid_segments_bool": valid_segments,
    }
    array_metadata = {}
    for name, array in arrays.items():
        path = out_dir / f"{name}.npy"
        tmp = out_dir / f".{name}.tmp.npy"
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        tmp.replace(path)
        array_metadata[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    summary = {
        "schema_version": 1,
        "status": "complete" if args.max_samples is None else "debug_subset",
        "purpose": "mira_style_temporal_cs2_action_recoverability_ratio",
        "definition": (
            "per-class generated-video action AP divided by model-native "
            "ground-truth-video action AP"
        ),
        "backbone_deviation_from_mira": (
            "MIRA uses gated DINOv3-B; this reproducible public evaluator uses "
            "commit-pinned DINOv2-B/14."
        ),
        "archive_metadata": str(args.archive_dir / "metadata.json"),
        "archive_metadata_sha256": sha256_file(args.archive_dir / "metadata.json"),
        "sample_plan_sha256": sha256_file(sample_plan_path),
        "probe_checkpoint": str(args.probe_checkpoint),
        "probe_checkpoint_sha256": sha256_file(args.probe_checkpoint),
        "backbone": actual_backbone,
        "num_samples": num_samples,
        "num_complete_segments": int(valid_segments.sum()),
        "num_incomplete_segments_excluded": int(
            valid_segments.size - valid_segments.sum()
        ),
        "eval_seeds": eval_seeds,
        "action_modes": action_modes,
        "label_names": list(ACTION_LABEL_NAMES),
        "segment_contract": {
            "frames": 5,
            "transitions": 4,
            "seconds_at_8fps": 0.5,
            "buttons": "active in any transition",
            "mouse": "signed summed degrees with absolute threshold 1.0",
            "post_alive": "exclude any segment with an invalid target step",
        },
        "results": arr,
        "artifacts": array_metadata,
        "deterministic_algorithms": True,
    }
    summary_path = out_dir / "summary.json"
    tmp_summary = out_dir / ".summary.tmp.json"
    tmp_summary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_summary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
