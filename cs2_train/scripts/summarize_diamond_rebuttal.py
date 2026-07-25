"""Audit and summarize the completed matched-arm DIAMOND rebuttal experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METRICS = (
    "denoising_loss",
    "one_step_mse",
    "rollout_mse_mean",
    "rollout_mse_last",
)
WINDOW_MODES = ("midpoint", "first-death")
TRAINING_ARMS = ("true", "shuffled")
ACTION_MODES = ("true", "shuffled", "zeros")
TRAINING_COMMIT = "34524f6b6f1f805200d72ab4e77f3a55dd6415f8"
CROSS_WINDOW_IDENTICAL_FIELDS = (
    "config_sha256",
    "manifest_sha256",
    "split",
    "map_slug",
    "target_fps",
    "resize",
    "rollout_steps",
    "rollout_target_masking",
    "num_denoising_steps",
    "s_cond",
    "eval_seeds",
    "action_modes",
    "num_eval_samples",
    "num_rounds",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clustered_summary(
    deltas: list[tuple[str, float]],
    *,
    bootstrap_seed: int,
    n_bootstrap: int = 10_000,
) -> dict:
    by_round: dict[str, list[float]] = defaultdict(list)
    for round_id, delta in deltas:
        by_round[round_id].append(float(delta))
    round_means = np.asarray(
        [np.mean(values) for values in by_round.values()],
        dtype=np.float64,
    )
    if not round_means.size:
        raise ValueError("paired contrast has no round clusters")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        round_means.size,
        size=(n_bootstrap, round_means.size),
    )
    bootstrap = round_means[indices].mean(axis=1)
    return {
        "mean": float(round_means.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "cluster_unit": "round_id",
        "num_round_clusters": int(round_means.size),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": n_bootstrap,
        "fraction_rounds_positive": float((round_means > 0).mean()),
    }


def row_map(
    rows: list[dict], metric: str
) -> dict[tuple[int, int, str], dict[str, object]]:
    mapped: dict[tuple[int, int, str], dict[str, object]] = {}
    for row in rows:
        key = (
            int(row["eval_seed"]),
            int(row["dataset_index"]),
            str(row["action_mode"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate evaluator row: {key}")
        mapped[key] = {
            "value": float(row[metric]),
            "round_id": str(row["round_id"]),
            "sample_key": str(row["sample_key"]),
        }
    return mapped


def validate_contract(
    summaries: dict[str, dict],
    *,
    expected_step: int,
    expected_samples: int,
) -> dict:
    first = next(iter(summaries.values()))
    identical_fields = (
        "config_sha256",
        "manifest_sha256",
        "split",
        "map_slug",
        "target_fps",
        "resize",
        "rollout_steps",
        "rollout_target_masking",
        "num_denoising_steps",
        "s_cond",
        "eval_seeds",
        "action_modes",
        "num_eval_samples",
        "num_rounds",
        "sample_plan_sha256",
        "action_plan_sha256",
    )
    for name, summary in summaries.items():
        if int(summary["checkpoint_step"]) != expected_step:
            raise ValueError(
                f"{name}: checkpoint step {summary['checkpoint_step']} != {expected_step}"
            )
        if int(summary["num_eval_samples"]) != expected_samples:
            raise ValueError(
                f"{name}: eval samples {summary['num_eval_samples']} != {expected_samples}"
            )
        for field in identical_fields:
            if summary[field] != first[field]:
                raise ValueError(
                    f"{name}: contract field {field} differs: "
                    f"{summary[field]!r} != {first[field]!r}"
                )
    if first["split"] != "test" or first["map_slug"] != "dust2":
        raise ValueError("confirmatory summary is not the Dust2 test split")
    return {field: first[field] for field in identical_fields}


def validate_window_contracts(contracts: dict[str, dict]) -> dict:
    """Require all non-window-specific evaluation settings to be identical.

    Midpoint and first-death deliberately have different sample and action
    plans, so their plan hashes are retained per window but excluded from this
    shared contract.
    """

    if set(contracts) != set(WINDOW_MODES):
        raise ValueError(
            f"window contracts {sorted(contracts)} != {sorted(WINDOW_MODES)}"
        )
    reference_mode = WINDOW_MODES[0]
    reference = contracts[reference_mode]
    for window_mode in WINDOW_MODES[1:]:
        candidate = contracts[window_mode]
        for field in CROSS_WINDOW_IDENTICAL_FIELDS:
            if candidate[field] != reference[field]:
                raise ValueError(
                    f"window modes differ on {field}: "
                    f"{reference_mode}={reference[field]!r} != "
                    f"{window_mode}={candidate[field]!r}"
                )
    return {field: reference[field] for field in CROSS_WINDOW_IDENTICAL_FIELDS}


def validate_training_runs(
    run_root: Path,
    summaries: dict[str, dict],
    *,
    expected_training_commit: str,
) -> dict:
    recorded_commit = (
        run_root / "provenance" / "training_commit.txt"
    ).read_text(encoding="utf-8").strip()
    if recorded_commit != expected_training_commit:
        raise ValueError(
            f"training commit {recorded_commit!r} != {expected_training_commit!r}"
        )

    configs = {}
    checkpoint_hashes = {}
    for arm in TRAINING_ARMS:
        config = json.loads(
            (run_root / arm / "config.json").read_text(encoding="utf-8")
        )
        if config.get("action_mode") != arm:
            raise ValueError(
                f"{arm}: recorded action mode {config.get('action_mode')!r}"
            )
        configs[arm] = config

        arm_summaries = [
            summaries[f"{arm}/{window_mode}"] for window_mode in WINDOW_MODES
        ]
        arm_hashes = {summary["checkpoint_sha256"] for summary in arm_summaries}
        if len(arm_hashes) != 1:
            raise ValueError(f"{arm}: window evaluations use different checkpoints")
        checkpoint_hashes[arm] = next(iter(arm_hashes))
        expected_checkpoint = (run_root / arm / "latest.pt").resolve()
        for summary in arm_summaries:
            if Path(summary["checkpoint"]).resolve() != expected_checkpoint:
                raise ValueError(
                    f"{arm}: evaluator did not use {expected_checkpoint}"
                )

    if checkpoint_hashes["true"] == checkpoint_hashes["shuffled"]:
        raise ValueError("training arms unexpectedly use the same checkpoint")
    if configs["true"]["baseline_config"] != configs["shuffled"]["baseline_config"]:
        raise ValueError("training arms use different frozen baseline configs")

    comparable_fields = (
        "seed",
        "action_shuffle_seed",
        "max_steps",
        "batch_size",
        "grad_acc",
        "target_fps",
        "manifest_name",
        "map_slug",
        "resolution",
        "preset",
        "resize",
        "num_autoregressive_steps",
        "lr",
        "lr_warmup",
        "weight_decay",
        "ema_decay",
        "mixed_precision",
        "deterministic",
        "val_every",
        "rollout_every",
        "rollout_steps",
        "val_batch_size",
        "val_denoise_steps",
        "val_s_cond",
        "use_ema_for_val",
    )
    for field in comparable_fields:
        if configs["true"].get(field) != configs["shuffled"].get(field):
            raise ValueError(
                f"training arms differ on {field}: "
                f"{configs['true'].get(field)!r} != "
                f"{configs['shuffled'].get(field)!r}"
            )
    return {
        "training_commit": recorded_commit,
        "action_mode_by_arm": {
            arm: configs[arm]["action_mode"] for arm in TRAINING_ARMS
        },
        "matched_training_fields": {
            field: configs["true"].get(field) for field in comparable_fields
        },
        "checkpoint_sha256_by_arm": checkpoint_hashes,
    }


def load_inline_trajectory(
    run_root: Path,
    *,
    expected_step: int,
    cadence: int,
) -> dict[str, list[dict]]:
    if expected_step <= 0 or cadence <= 0 or expected_step % cadence:
        raise ValueError("endpoint must be a positive multiple of trajectory cadence")
    expected_steps = list(range(cadence, expected_step + 1, cadence))
    output = {}
    for arm in TRAINING_ARMS:
        rows = load_jsonl(run_root / arm / "metrics.jsonl")
        by_key: dict[tuple[str, int], dict] = {}
        duplicates: dict[tuple[str, int], int] = defaultdict(int)
        for row in rows:
            kind = str(row.get("kind", ""))
            if kind not in {"validation", "rollout"}:
                continue
            key = (kind, int(row["step"]))
            duplicates[key] += 1
            if key in by_key and row != by_key[key]:
                raise ValueError(f"{arm}: conflicting duplicate inline metric {key}")
            by_key[key] = row

        arm_steps = sorted({step for _kind, step in by_key})
        if arm_steps != expected_steps:
            raise ValueError(
                f"{arm}: inline evaluation steps {arm_steps} != {expected_steps}"
            )
        trajectory = []
        for step in expected_steps:
            validation = by_key[("validation", step)]
            rollout = by_key[("rollout", step)]
            trajectory.append(
                {
                    "step": step,
                    "weights": validation["weights"],
                    "one_step": {
                        "true_mse": float(validation["true"]["val_mse"]),
                        "shuffled_mse": float(validation["shuffled"]["val_mse"]),
                        "shuffled_minus_true_mse": float(
                            validation["shuffled_minus_true_mse"]
                        ),
                    },
                    "rollout": {
                        "true_mse_mean": float(rollout["true_mse_mean"]),
                        "shuffled_mse_mean": float(rollout["shuffled_mse_mean"]),
                        "shuffled_minus_true_mse_mean": float(
                            rollout["shuffled_minus_true_mse_mean"]
                        ),
                        "true_mse_per_step": [
                            float(value)
                            for value in rollout["true"]["rollout_mse_per_step"]
                        ],
                        "shuffled_mse_per_step": [
                            float(value)
                            for value in rollout["shuffled"][
                                "rollout_mse_per_step"
                            ]
                        ],
                    },
                    "duplicate_records": {
                        kind: duplicates[(kind, step)]
                        for kind in ("validation", "rollout")
                    },
                }
            )
        output[arm] = trajectory
    return output


def summarize_window(
    *,
    true_summary: dict,
    shuffled_summary: dict,
    true_rows: list[dict],
    shuffled_rows: list[dict],
    bootstrap_seed: int,
) -> dict:
    def rollout_validity(rows: list[dict]) -> dict | None:
        rows_with_masks = [
            row
            for row in rows
            if "rollout_valid_steps" in row and "rollout_valid_count" in row
        ]
        if not rows_with_masks:
            return None
        masks_by_sample: dict[int, tuple[bool, ...]] = {}
        for row in rows_with_masks:
            dataset_index = int(row["dataset_index"])
            mask = tuple(bool(value) for value in row["rollout_valid_steps"])
            prior = masks_by_sample.setdefault(dataset_index, mask)
            if prior != mask:
                raise ValueError(
                    f"rollout validity mask differs for dataset row {dataset_index}"
                )
            if int(row["rollout_valid_count"]) != sum(mask):
                raise ValueError(
                    f"rollout valid-count mismatch for dataset row {dataset_index}"
                )
        counts = Counter(sum(mask) for mask in masks_by_sample.values())
        rollout_steps = len(next(iter(masks_by_sample.values())))
        valid_steps = sum(count * frequency for count, frequency in counts.items())
        return {
            "num_eval_samples": len(masks_by_sample),
            "rollout_steps": rollout_steps,
            "valid_count_distribution": {
                str(count): int(frequency) for count, frequency in sorted(counts.items())
            },
            "valid_target_steps": int(valid_steps),
            "planned_target_steps": len(masks_by_sample) * rollout_steps,
            "masked_post_alive_target_steps": (
                len(masks_by_sample) * rollout_steps - int(valid_steps)
            ),
            "one_step_targets_all_valid": all(mask[0] for mask in masks_by_sample.values()),
        }

    validity_by_arm = {
        "true": rollout_validity(true_rows),
        "shuffled": rollout_validity(shuffled_rows),
    }
    if validity_by_arm["true"] != validity_by_arm["shuffled"]:
        raise ValueError("training arms use different rollout target-validity masks")
    output: dict[str, dict] = {}
    if validity_by_arm["true"] is not None:
        output["rollout_target_validity"] = validity_by_arm["true"]
    for metric_index, metric in enumerate(METRICS):
        maps = {
            "true": row_map(true_rows, metric),
            "shuffled": row_map(shuffled_rows, metric),
        }
        if set(maps["true"]) != set(maps["shuffled"]):
            raise ValueError(
                f"{metric}: checkpoint arms do not have identical evaluator rows"
            )

        metric_output = {
            "checkpoint_means": {
                arm: {
                    mode: float(summary["means"][mode][metric]) for mode in ACTION_MODES
                }
                for arm, summary in (
                    ("true", true_summary),
                    ("shuffled", shuffled_summary),
                )
            },
            "within_checkpoint_action_sensitivity": {},
            "training_effect": {},
        }
        sensitivity_values: dict[str, dict[tuple[int, int], tuple[str, float]]] = {}
        for arm in TRAINING_ARMS:
            mapped = maps[arm]
            arm_sensitivity: dict[tuple[int, int], tuple[str, float]] = {}
            deltas = []
            base_keys = {
                (eval_seed, dataset_index)
                for eval_seed, dataset_index, mode in mapped
                if mode == "true"
            }
            for eval_seed, dataset_index in sorted(base_keys):
                true_item = mapped[(eval_seed, dataset_index, "true")]
                shuffled_item = mapped[(eval_seed, dataset_index, "shuffled")]
                if (
                    true_item["round_id"] != shuffled_item["round_id"]
                    or true_item["sample_key"] != shuffled_item["sample_key"]
                ):
                    raise ValueError("action modes are not sample-paired")
                delta = float(shuffled_item["value"]) - float(true_item["value"])
                round_id = str(true_item["round_id"])
                arm_sensitivity[(eval_seed, dataset_index)] = (round_id, delta)
                deltas.append((round_id, delta))
            sensitivity_values[arm] = arm_sensitivity
            metric_output["within_checkpoint_action_sensitivity"][arm] = (
                clustered_summary(
                    deltas,
                    bootstrap_seed=bootstrap_seed
                    + metric_index * 100
                    + (0 if arm == "true" else 1),
                )
            )

        for mode_index, mode in enumerate(ACTION_MODES):
            deltas = []
            for key in sorted(maps["true"]):
                if key[2] != mode:
                    continue
                true_item = maps["true"][key]
                shuffled_item = maps["shuffled"][key]
                if true_item["sample_key"] != shuffled_item["sample_key"]:
                    raise ValueError("training arms are not sample-paired")
                # Positive means aligned training has lower error.
                deltas.append(
                    (
                        str(true_item["round_id"]),
                        float(shuffled_item["value"]) - float(true_item["value"]),
                    )
                )
            metric_output["training_effect"][mode] = clustered_summary(
                deltas,
                bootstrap_seed=bootstrap_seed + metric_index * 100 + 10 + mode_index,
            )

        did = []
        if set(sensitivity_values["true"]) != set(sensitivity_values["shuffled"]):
            raise ValueError("action-sensitivity arms are not paired")
        for key in sorted(sensitivity_values["true"]):
            true_round, true_delta = sensitivity_values["true"][key]
            shuffled_round, shuffled_delta = sensitivity_values["shuffled"][key]
            if true_round != shuffled_round:
                raise ValueError("difference-in-differences round mismatch")
            # Positive means aligned training increased action sensitivity.
            did.append((true_round, true_delta - shuffled_delta))
        metric_output["action_sensitivity_difference_in_differences"] = (
            clustered_summary(
                did,
                bootstrap_seed=bootstrap_seed + metric_index * 100 + 20,
            )
        )
        output[metric] = metric_output
    return output


def render_markdown(summary: dict) -> str:
    lines = [
        "# DIAMOND CounterStrike-1K Dust2 confirmatory summary",
        "",
        f"- Endpoint: step {summary['endpoint_step']:,} for both training arms",
        f"- Test samples: {summary['contract']['num_eval_samples']:,} POV rows",
        f"- Test rounds: {summary['contract']['num_rounds']}",
        f"- Eval seeds: {summary['contract']['eval_seeds']}",
        "",
        "Positive deltas favor aligned actions/training under the definitions below.",
        "",
    ]
    for window_mode, window in summary["windows"].items():
        validity = window.get("rollout_target_validity")
        lines.extend(
            [
                f"## {window_mode}",
                "",
                *(
                    [
                        (
                            f"- Valid rollout targets: {validity['valid_target_steps']:,} / "
                            f"{validity['planned_target_steps']:,}; masked post-alive targets: "
                            f"{validity['masked_post_alive_target_steps']:,}"
                        ),
                        f"- Valid-step count distribution: {validity['valid_count_distribution']}",
                        "",
                    ]
                    if validity is not None
                    else []
                ),
                "| Metric | True-trained action sensitivity | Shuffled-trained action sensitivity | Difference in differences | True-action training effect |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in METRICS:
            payload = window[metric]
            true_sensitivity = payload["within_checkpoint_action_sensitivity"]["true"]
            shuffled_sensitivity = payload["within_checkpoint_action_sensitivity"][
                "shuffled"
            ]
            did = payload["action_sensitivity_difference_in_differences"]
            training_effect = payload["training_effect"]["true"]

            def cell(value: dict) -> str:
                return (
                    f"{value['mean']:.6f} "
                    f"[{value['ci95'][0]:.6f}, {value['ci95'][1]:.6f}]"
                )

            lines.append(
                f"| {metric} | {cell(true_sensitivity)} | "
                f"{cell(shuffled_sensitivity)} | {cell(did)} | "
                f"{cell(training_effect)} |"
            )
        lines.append("")
    lines.extend(
        [
            "Definitions:",
            "",
            "- action sensitivity = MSE(shuffled input) - MSE(true input);",
            "- difference in differences = action sensitivity of true-trained checkpoint - action sensitivity of shuffled-trained checkpoint;",
            "- true-action training effect = MSE(shuffled-trained checkpoint) - MSE(true-trained checkpoint), both evaluated with true actions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-step", type=int, default=50_000)
    parser.add_argument("--expected-samples", type=int, default=690)
    parser.add_argument("--bootstrap-seed", type=int, default=20250725)
    parser.add_argument(
        "--expected-training-commit",
        default=TRAINING_COMMIT,
    )
    args = parser.parse_args()

    summaries: dict[str, dict] = {}
    rows: dict[str, list[dict]] = {}
    for arm in TRAINING_ARMS:
        for window_mode in WINDOW_MODES:
            key = f"{arm}/{window_mode}"
            evaluation = args.run_root / arm / "evaluation" / window_mode
            summaries[key] = json.loads(
                (evaluation / "summary.json").read_text(encoding="utf-8")
            )
            rows[key] = load_jsonl(evaluation / "per_sample_metrics.jsonl")

    training_audit = validate_training_runs(
        args.run_root,
        summaries,
        expected_training_commit=args.expected_training_commit,
    )
    inline_trajectory = load_inline_trajectory(
        args.run_root,
        expected_step=args.expected_step,
        cadence=int(training_audit["matched_training_fields"]["val_every"]),
    )
    contracts = {}
    windows = {}
    for window_index, window_mode in enumerate(WINDOW_MODES):
        window_summaries = {
            arm: summaries[f"{arm}/{window_mode}"] for arm in TRAINING_ARMS
        }
        window_contract = validate_contract(
            window_summaries,
            expected_step=args.expected_step,
            expected_samples=args.expected_samples,
        )
        contracts[window_mode] = window_contract
        windows[window_mode] = summarize_window(
            true_summary=window_summaries["true"],
            shuffled_summary=window_summaries["shuffled"],
            true_rows=rows[f"true/{window_mode}"],
            shuffled_rows=rows[f"shuffled/{window_mode}"],
            bootstrap_seed=args.bootstrap_seed + window_index * 10_000,
        )
    shared_contract = validate_window_contracts(contracts)

    output = {
        "schema_version": 2,
        "status": "complete",
        "endpoint_step": args.expected_step,
        "training_audit": training_audit,
        "inline_validation_trajectory": {
            "status": "diagnostic_only",
            "rng_contract": (
                "true and shuffled actions share diffusion draws within each "
                "checkpoint; the deterministic seed changes with checkpoint step"
            ),
            "arms": inline_trajectory,
        },
        "contract": shared_contract,
        "contract_by_window": contracts,
        "checkpoint_sha256": {
            key: summary["checkpoint_sha256"] for key, summary in summaries.items()
        },
        "windows": windows,
    }
    output_dir = args.run_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rebuttal_summary.json"
    markdown_path = output_dir / "rebuttal_summary.md"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(output), encoding="utf-8")
    print(
        json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2)
    )


if __name__ == "__main__":
    main()
