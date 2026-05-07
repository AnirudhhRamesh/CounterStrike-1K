"""Run the CounterStrike-1K video consistency evaluation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cs2_release.encoders.registry import canonical_encoder_name
from cs2_release.core.tracking import add_wandb_args


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def wandb_args(args, *, stage: str) -> list[str]:
    if not args.wandb_project or args.wandb_mode == "disabled":
        return []
    out = ["--wandb-project", args.wandb_project, "--wandb-mode", args.wandb_mode]
    if args.wandb_entity:
        out += ["--wandb-entity", args.wandb_entity]
    group = args.wandb_group or args.out.name
    out += ["--wandb-group", group]
    base_name = args.wandb_run_name or args.out.name
    out += ["--wandb-run-name", f"{base_name}-{stage}"]
    tags = args.wandb_tags
    if tags:
        out += ["--wandb-tags", tags]
    if args.wandb_log_artifacts:
        out += ["--wandb-log-artifacts"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoder", default="rgb_hist")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--windows-per-round", type=int, default=1)
    parser.add_argument("--max-rounds-per-split", type=int, default=500)
    parser.add_argument("--retrieval-candidates", type=int, default=32)
    parser.add_argument("--max-retrieval-queries", type=int, default=5000)
    parser.add_argument("--retrieval-negative-policies", nargs="+",
                        default=["same_map_phase_different_round"])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-extended", action="store_true",
                        help="Run multi-positive, spatial, action, k-POV, global-state, off-screen, split hygiene, and qualitative evals.")
    parser.add_argument("--skip-corruption", action="store_true")
    parser.add_argument("--corruption-negatives-per-positive", type=int, default=1)
    parser.add_argument("--corruption-epochs", type=int, default=20)
    parser.add_argument("--action-probe-epochs", type=int, default=40)
    parser.add_argument("--multipov-probe-epochs", type=int, default=60)
    parser.add_argument("--global-probe-epochs", type=int, default=60)
    parser.add_argument("--offscreen-probe-epochs", type=int, default=60)
    parser.add_argument("--spatial-positive-radius", type=float, default=1200.0)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--skip-embedding-extraction", action="store_true",
                        help="Reuse embeddings/<encoder> under --out and run downstream evaluations only.")
    parser.add_argument("--seed", type=int, default=123)
    add_wandb_args(parser)
    args = parser.parse_args()

    py = sys.executable
    shard_root = args.shard_root or args.root
    args.out.mkdir(parents=True, exist_ok=True)
    windows = args.out / "windows" / "eval_windows.parquet"
    embeddings_dir = args.out / "embeddings" / canonical_encoder_name(args.encoder)
    retrieval_pairs = args.out / "retrieval" / "retrieval_eval.parquet"
    corruption_train = args.out / "corruption" / "corruption_train.parquet"
    corruption_val = args.out / "corruption" / "corruption_val.parquet"
    corruption_test = args.out / "corruption" / "corruption_test.parquet"
    corruption_ckpt_dir = args.out / "corruption" / "checkpoint"

    run([
        py, "-m", "cs2_release.core.windows",
        "--root", str(args.root),
        "--window-seconds", str(args.window_seconds),
        "--windows-per-round", str(args.windows_per_round),
        "--max-rounds-per-split", str(args.max_rounds_per_split),
        "--seed", str(args.seed),
        "--out", str(windows),
    ])
    if not args.skip_embedding_extraction:
        run([
            py, "-m", "cs2_release.encoders.extract_video",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(windows),
            "--encoder", args.encoder,
            "--device", args.device,
            "--frames-per-window", str(args.frames_per_window),
            "--out", str(args.out / "embeddings"),
        ] + wandb_args(args, stage="embeddings"))
    retrieval_dirs_by_policy: dict[str, Path] = {}
    for policy in args.retrieval_negative_policies:
        retrieval_dir = args.out / ("retrieval" if policy == args.retrieval_negative_policies[0] else f"retrieval_{policy}")
        pairs_path = retrieval_pairs if policy == args.retrieval_negative_policies[0] else retrieval_dir / "retrieval_eval.parquet"
        retrieval_dirs_by_policy[policy] = retrieval_dir
        run([
            py, "-m", "cs2_release.retrieval.pairs.basic",
            "--windows", str(windows),
            "--split", "test",
            "--candidates-per-query", str(args.retrieval_candidates),
            "--max-queries", str(args.max_retrieval_queries),
            "--negative-policy", policy,
            "--seed", str(args.seed),
            "--out", str(pairs_path),
        ])
        run([
            py, "-m", "cs2_release.retrieval.eval_basic",
            "--pairs", str(pairs_path),
            "--embeddings", str(embeddings_dir),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(retrieval_dir),
        ] + wandb_args(args, stage=f"retrieval-{policy}"))
    if args.run_extended:
        multi_dir = args.out / "multipositive_retrieval"
        spatial_dir = args.out / "spatial_retrieval"
        spatial_far_dir = args.out / "spatial_retrieval_same_time_far"
        spatial_location_dir = args.out / "spatial_retrieval_same_location_wrong_time"
        pairwise_spatial_dir = args.out / "pairwise_spatial_probe"
        action_dir = args.out / "action_probe"
        coverage_dir = args.out / "multipov_action_coverage"
        off_pov_dir = args.out / "off_pov_action_visibility"
        multipov_dir = args.out / "multipov_action_probe"
        global_event_dir = args.out / "global_event_probe"
        offscreen_probe_dir = args.out / "offscreen_action_probe"
        temporal_dir = args.out / "temporal_alignment"
        run([
            py, "-m", "cs2_release.retrieval.pairs.multipositive",
            "--windows", str(windows),
            "--split", "test",
            "--candidates-per-query", str(args.retrieval_candidates),
            "--max-queries", str(args.max_retrieval_queries),
            "--seed", str(args.seed),
            "--out", str(multi_dir / "retrieval_eval.parquet"),
        ])
        run([
            py, "-m", "cs2_release.retrieval.eval_multipositive",
            "--pairs", str(multi_dir / "retrieval_eval.parquet"),
            "--embeddings", str(embeddings_dir),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(multi_dir),
        ] + wandb_args(args, stage="multipositive-retrieval"))
        run([
            py, "-m", "cs2_release.retrieval.pairs.spatial",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(windows),
            "--split", "test",
            "--candidates-per-query", str(args.retrieval_candidates),
            "--max-queries", str(args.max_retrieval_queries),
            "--positive-radius", str(args.spatial_positive_radius),
            "--max-positives", "3",
            "--seed", str(args.seed),
            "--out", str(spatial_dir / "retrieval_eval.parquet"),
        ])
        run([
            py, "-m", "cs2_release.retrieval.eval_multipositive",
            "--pairs", str(spatial_dir / "retrieval_eval.parquet"),
            "--embeddings", str(embeddings_dir),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(spatial_dir),
        ] + wandb_args(args, stage="spatial-retrieval"))
        run([
            py, "-m", "cs2_release.retrieval.pairs.spatial",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(windows),
            "--split", "test",
            "--candidates-per-query", str(args.retrieval_candidates),
            "--max-queries", str(args.max_retrieval_queries),
            "--positive-radius", str(args.spatial_positive_radius),
            "--negative-policy", "same_time_far_location",
            "--max-positives", "3",
            "--seed", str(args.seed),
            "--out", str(spatial_far_dir / "retrieval_eval.parquet"),
        ])
        run([
            py, "-m", "cs2_release.retrieval.eval_multipositive",
            "--pairs", str(spatial_far_dir / "retrieval_eval.parquet"),
            "--embeddings", str(embeddings_dir),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(spatial_far_dir),
        ] + wandb_args(args, stage="spatial-same-time-far"))
        run([
            py, "-m", "cs2_release.retrieval.pairs.spatial",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(windows),
            "--split", "test",
            "--candidates-per-query", str(args.retrieval_candidates),
            "--max-queries", str(args.max_retrieval_queries),
            "--positive-radius", str(args.spatial_positive_radius),
            "--negative-policy", "same_location_wrong_time",
            "--negative-location-radius", str(args.spatial_positive_radius),
            "--max-positives", "3",
            "--seed", str(args.seed),
            "--out", str(spatial_location_dir / "retrieval_eval.parquet"),
        ])
        run([
            py, "-m", "cs2_release.retrieval.eval_multipositive",
            "--pairs", str(spatial_location_dir / "retrieval_eval.parquet"),
            "--embeddings", str(embeddings_dir),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(spatial_location_dir),
        ] + wandb_args(args, stage="spatial-same-location-wrong-time"))
        run([
            py, "-m", "cs2_release.retrieval.eval_pairwise_spatial",
            "--pairs", str(spatial_far_dir / "retrieval_eval.parquet"),
            "--embeddings", str(embeddings_dir),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(pairwise_spatial_dir),
        ] + wandb_args(args, stage="pairwise-spatial-probe"))
        run([
            py, "-m", "cs2_release.action_probe.labels",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(windows),
            "--out", str(action_dir / "action_probe_labels.parquet"),
        ])
        run([
            py, "-m", "cs2_release.action_probe.train_video",
            "--labels", str(action_dir / "action_probe_labels.parquet"),
            "--embeddings", str(embeddings_dir),
            "--epochs", str(args.action_probe_epochs),
            "--device", args.device,
            "--seed", str(args.seed),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(action_dir),
        ] + wandb_args(args, stage="action-probe"))
        run([
            py, "-m", "cs2_release.action_probe.coverage",
            "--labels", str(action_dir / "action_probe_labels.parquet"),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--seed", str(args.seed),
            "--out", str(coverage_dir),
        ])
        run([
            py, "-m", "cs2_release.action_probe.off_pov",
            "--labels", str(action_dir / "action_probe_labels.parquet"),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--seed", str(args.seed),
            "--out", str(off_pov_dir),
        ])
        run([
            py, "-m", "cs2_release.action_probe.train_multipov",
            "--labels", str(action_dir / "action_probe_labels.parquet"),
            "--embeddings", str(embeddings_dir),
            "--epochs", str(args.multipov_probe_epochs),
            "--device", args.device,
            "--seed", str(args.seed),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(multipov_dir),
        ] + wandb_args(args, stage="multipov-action-probe"))
        run([
            py, "-m", "cs2_release.global_state.train",
            "--root", str(args.root),
            "--windows", str(windows),
            "--embeddings", str(embeddings_dir),
            "--epochs", str(args.global_probe_epochs),
            "--device", args.device,
            "--seed", str(args.seed),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(global_event_dir),
        ] + wandb_args(args, stage="global-event-probe"))
        run([
            py, "-m", "cs2_release.action_probe.train_offscreen",
            "--labels", str(action_dir / "action_probe_labels.parquet"),
            "--embeddings", str(embeddings_dir),
            "--epochs", str(args.offscreen_probe_epochs),
            "--device", args.device,
            "--seed", str(args.seed),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--out", str(offscreen_probe_dir),
        ] + wandb_args(args, stage="offscreen-action-probe"))
        if "same_round_wrong_time" in retrieval_dirs_by_policy:
            run([
                py, "-m", "cs2_release.retrieval.eval_temporal",
                "--predictions", str(retrieval_dirs_by_policy["same_round_wrong_time"] / "retrieval_predictions.parquet"),
                "--bootstrap-samples", str(args.bootstrap_samples),
                "--seed", str(args.seed),
                "--out", str(temporal_dir),
            ])
        run([
            py, "-m", "cs2_release.hygiene.eval_split",
            "--root", str(args.root),
            "--windows", str(windows),
            "--out", str(args.out / "split_hygiene" / "metrics_split_hygiene.json"),
        ])
        run([
            py, "-m", "cs2_release.retrieval.render_qualitative",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(windows),
            "--predictions", str(spatial_dir / "retrieval_predictions.parquet"),
            "--labels", str(action_dir / "action_probe_labels.parquet"),
            "--out", str(args.out / "qualitative" / "spatial_retrieval_examples.png"),
        ])
    if args.skip_corruption:
        return 0
    for split, out_path in (
        ("train", corruption_train),
        ("val", corruption_val),
        ("test", corruption_test),
    ):
        run([
            py, "-m", "cs2_release.corruption.make_packs",
            "--windows", str(windows),
            "--split", split,
            "--negatives-per-positive", str(args.corruption_negatives_per_positive),
            "--seed", str(args.seed),
            "--out", str(out_path),
        ])
    run([
        py, "-m", "cs2_release.corruption.train",
        "--train-packs", str(corruption_train),
        "--val-packs", str(corruption_val),
        "--embeddings", str(embeddings_dir),
        "--epochs", str(args.corruption_epochs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--out", str(corruption_ckpt_dir),
    ] + wandb_args(args, stage="corruption-train"))
    run([
        py, "-m", "cs2_release.corruption.eval",
        "--packs", str(corruption_test),
        "--embeddings", str(embeddings_dir),
        "--checkpoint", str(corruption_ckpt_dir / "corruption_head.pt"),
        "--device", args.device,
        "--out", str(args.out / "corruption" / "test"),
    ] + wandb_args(args, stage="corruption-test"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
