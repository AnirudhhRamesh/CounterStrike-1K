"""Run and summarize a fixed-noise DIAMOND validation checkpoint audit.

This is a secondary convergence diagnostic used only to decide whether a
matched post-50k extension is warranted. It never reads the test split.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from cs2_train.scripts.summarize_diamond_rebuttal import (
    METRICS,
    TRAINING_ARMS,
    clustered_summary,
    load_jsonl,
)

DEFAULT_STEPS = (10_000, 20_000, 30_000, 40_000, 50_000)
EVAL_SEEDS = (37, 41, 43)


def evaluator_command(
    *,
    python: str,
    config: Path,
    data_dir: Path,
    checkpoint: Path,
    out_dir: Path,
    pov_idx: int,
    expected_samples: int,
) -> list[str]:
    return [
        python,
        "-m",
        "cs2_train.src.evaluate_action_sensitivity",
        "--config",
        str(config),
        "--data-dir",
        str(data_dir),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
        "--split",
        "val",
        "--window-mode",
        "midpoint",
        "--pov-idx",
        str(pov_idx),
        "--expected-samples",
        str(expected_samples),
        "--eval-seeds",
        *(str(seed) for seed in EVAL_SEEDS),
        "--max-review-videos",
        "0",
    ]


def validate_summary(
    summary: dict,
    *,
    checkpoint_step: int,
    pov_idx: int,
    expected_samples: int,
) -> None:
    expected = {
        "checkpoint_step": checkpoint_step,
        "split": "val",
        "map_slug": "dust2",
        "window_mode": "midpoint",
        "pov_idx_filter": pov_idx,
        "num_eval_samples": expected_samples,
        "num_rounds": expected_samples,
        "eval_seeds": list(EVAL_SEEDS),
        "action_modes": ["true", "shuffled", "zeros"],
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(
                f"step {checkpoint_step}: {field}={summary.get(field)!r} != {value!r}"
            )


def action_sensitivity_by_key(
    rows: list[dict],
    metric: str,
) -> dict[tuple[int, int], tuple[str, float]]:
    modes = {}
    for row in rows:
        key = (int(row["eval_seed"]), int(row["dataset_index"]))
        modes.setdefault(key, {})[str(row["action_mode"])] = row
    output = {}
    for key, values in modes.items():
        if "true" not in values or "shuffled" not in values:
            raise ValueError(f"{metric}: incomplete true/shuffled pair at {key}")
        true = values["true"]
        shuffled = values["shuffled"]
        if (
            true["round_id"] != shuffled["round_id"]
            or true["sample_key"] != shuffled["sample_key"]
        ):
            raise ValueError(f"{metric}: evaluator pair mismatch at {key}")
        output[key] = (
            str(true["round_id"]),
            float(shuffled[metric]) - float(true[metric]),
        )
    return output


def endpoint_change(
    old_rows: list[dict],
    new_rows: list[dict],
    *,
    metric: str,
    bootstrap_seed: int,
) -> dict:
    old = action_sensitivity_by_key(old_rows, metric)
    new = action_sensitivity_by_key(new_rows, metric)
    if set(old) != set(new):
        raise ValueError(f"{metric}: checkpoint evaluator rows differ")
    deltas = []
    for key in sorted(old):
        old_round, old_value = old[key]
        new_round, new_value = new[key]
        if old_round != new_round:
            raise ValueError(f"{metric}: checkpoint round mismatch at {key}")
        deltas.append((old_round, new_value - old_value))
    result = clustered_summary(
        deltas,
        bootstrap_seed=bootstrap_seed,
    )
    low, high = result["ci95"]
    result["delta_definition"] = "sensitivity_at_new_step_minus_old_step"
    result["trend"] = (
        "increasing" if low > 0 else "decreasing" if high < 0 else "inconclusive"
    )
    return result


def render_markdown(payload: dict) -> str:
    lines = [
        "# DIAMOND validation checkpoint audit",
        "",
        "- Split: validation only; the test split is not read",
        f"- POV slot: {payload['contract']['pov_idx']}",
        f"- Validation rounds: {payload['contract']['expected_samples']}",
        f"- Evaluation seeds: {payload['contract']['eval_seeds']}",
        "- Diffusion draws, sample plan, and action-donor plan are fixed across checkpoints",
        "",
    ]
    for arm in TRAINING_ARMS:
        lines.extend(
            [
                f"## {arm}-trained",
                "",
                "| Step | One-step true MSE | One-step action delta | Rollout true MSE | Rollout action delta |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for point in payload["arms"][arm]:
            one = point["means"]["true"]["one_step_mse"]
            one_delta = point["paired_deltas"]["one_step_mse"]["shuffled"][
                "mean_delta"
            ]
            rollout = point["means"]["true"]["rollout_mse_mean"]
            rollout_delta = point["paired_deltas"]["rollout_mse_mean"]["shuffled"][
                "mean_delta"
            ]
            lines.append(
                f"| {point['step']:,} | {one:.6f} | {one_delta:+.6f} | "
                f"{rollout:.6f} | {rollout_delta:+.6f} |"
            )
        lines.extend(
            [
                "",
                "Endpoint sensitivity change (50k minus 40k):",
                "",
                "| Metric | Mean change | 95% round-clustered CI | Trend |",
                "|---|---:|---:|---|",
            ]
        )
        for metric in METRICS:
            result = payload["endpoint_sensitivity_change"][arm][metric]
            lines.append(
                f"| {metric} | {result['mean']:+.6f} | "
                f"[{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}] | "
                f"{result['trend']} |"
            )
        lines.append("")
    lines.extend(
        [
            "A positive action delta means shuffled inputs have higher error than",
            "the correct action. A positive endpoint change means action sensitivity",
            "increased from step 40,000 to 50,000. This validation-only diagnostic",
            "does not alter the preregistered step-50,000 test endpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--pov-idx", type=int, default=0)
    parser.add_argument("--expected-samples", type=int, default=54)
    parser.add_argument("--bootstrap-seed", type=int, default=20250726)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    steps = sorted(set(args.steps))
    if len(steps) < 2 or steps[-2:] != [40_000, 50_000]:
        raise ValueError("checkpoint audit must include steps 40,000 and 50,000")
    config = args.config or args.run_root / "provenance" / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)

    summaries = {}
    rows = {}
    for arm in TRAINING_ARMS:
        for step in steps:
            checkpoint = args.run_root / arm / f"step_{step:07d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            out_dir = (
                args.run_root
                / arm
                / "evaluation"
                / "validation-checkpoints"
                / f"step_{step:07d}"
            )
            summary_path = out_dir / "summary.json"
            command = evaluator_command(
                python=args.python,
                config=config,
                data_dir=args.data_dir,
                checkpoint=checkpoint,
                out_dir=out_dir,
                pov_idx=args.pov_idx,
                expected_samples=args.expected_samples,
            )
            if args.force or not summary_path.is_file():
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "command.sh").write_text(
                    shlex.join(command) + "\n",
                    encoding="utf-8",
                )
                with (out_dir / "evaluator.log").open("a", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_summary(
                summary,
                checkpoint_step=step,
                pov_idx=args.pov_idx,
                expected_samples=args.expected_samples,
            )
            key = (arm, step)
            summaries[key] = summary
            rows[key] = load_jsonl(out_dir / "per_sample_metrics.jsonl")

    contract_fields = (
        "config_sha256",
        "manifest_sha256",
        "sample_plan_sha256",
        "action_plan_sha256",
        "resize",
        "target_fps",
        "rollout_steps",
        "num_denoising_steps",
        "s_cond",
        "eval_seeds",
        "action_modes",
        "num_eval_samples",
        "num_rounds",
        "pov_idx_filter",
    )
    first = summaries[(TRAINING_ARMS[0], steps[0])]
    for key, summary in summaries.items():
        for field in contract_fields:
            if summary[field] != first[field]:
                raise ValueError(
                    f"{key}: fixed validation contract differs on {field}"
                )

    arms = {
        arm: [
            {
                "step": step,
                "checkpoint_sha256": summaries[(arm, step)]["checkpoint_sha256"],
                "means": summaries[(arm, step)]["means"],
                "paired_deltas": summaries[(arm, step)]["paired_deltas"],
            }
            for step in steps
        ]
        for arm in TRAINING_ARMS
    }
    endpoint_sensitivity_change = {}
    for arm_index, arm in enumerate(TRAINING_ARMS):
        endpoint_sensitivity_change[arm] = {
            metric: endpoint_change(
                rows[(arm, 40_000)],
                rows[(arm, 50_000)],
                metric=metric,
                bootstrap_seed=args.bootstrap_seed
                + arm_index * 1_000
                + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }

    payload = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "secondary_validation_only_convergence_audit",
        "contract": {
            "split": "val",
            "map_slug": "dust2",
            "window_mode": "midpoint",
            "pov_idx": args.pov_idx,
            "expected_samples": args.expected_samples,
            "eval_seeds": list(EVAL_SEEDS),
            "checkpoints": steps,
            "fixed_noise_across_checkpoints": True,
            "sample_plan_sha256": first["sample_plan_sha256"],
            "action_plan_sha256": first["action_plan_sha256"],
            "manifest_sha256": first["manifest_sha256"],
            "config_sha256": first["config_sha256"],
        },
        "arms": arms,
        "endpoint_sensitivity_change": endpoint_sensitivity_change,
    }
    output_dir = args.run_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_checkpoint_trajectory.json"
    markdown_path = output_dir / "validation_checkpoint_trajectory.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
