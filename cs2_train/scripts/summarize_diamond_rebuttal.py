"""Audit and summarize the completed matched-arm DIAMOND rebuttal experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
TRAINING_COMMIT = "a5566a05292088b0a5ac108388f90d074890d278"


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


def summarize_window(
    *,
    true_summary: dict,
    shuffled_summary: dict,
    true_rows: list[dict],
    shuffled_rows: list[dict],
    bootstrap_seed: int,
) -> dict:
    output: dict[str, dict] = {}
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
        lines.extend(
            [
                f"## {window_mode}",
                "",
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
    contract = {}
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
        if (
            contract
            and window_contract["manifest_sha256"] != contract["manifest_sha256"]
        ):
            raise ValueError("window modes use different manifests")
        contract = window_contract
        windows[window_mode] = summarize_window(
            true_summary=window_summaries["true"],
            shuffled_summary=window_summaries["shuffled"],
            true_rows=rows[f"true/{window_mode}"],
            shuffled_rows=rows[f"shuffled/{window_mode}"],
            bootstrap_seed=args.bootstrap_seed + window_index * 10_000,
        )

    output = {
        "schema_version": 1,
        "status": "complete",
        "endpoint_step": args.expected_step,
        "training_audit": training_audit,
        "contract": contract,
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
