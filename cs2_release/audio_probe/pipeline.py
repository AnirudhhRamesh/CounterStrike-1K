"""End-to-end runner for the CounterStrike-1K audio action probe.

Pipeline:

1. ``cs2_release.core.windows`` — synchronized 10-POV eval windows.
2. ``cs2_release.action_probe.labels`` — per-window multi-label action targets
   (FIRE, RIGHTCLICK, RELOAD, ...). Identical to the video probe inputs.
3. ``cs2_release.encoders.extract_audio`` — log-mel spectrograms decoded from the
   release MP4 audio tracks (stereo AAC -> mono 16 kHz).
4. ``cs2_release.audio_probe.train`` — small Conv2d -> GAP -> MLP probe trained
   end-to-end on the log-mel features with cluster-bootstrapped CIs.

The output directory mirrors the video probe layout so the resulting
``metrics_audio_probe.json`` is directly comparable to
``metrics_action_probe.json`` window-for-window.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
    if args.wandb_tags:
        out += ["--wandb-tags", args.wandb_tags]
    if args.wandb_log_artifacts:
        out += ["--wandb-log-artifacts"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Local CounterStrike-1K release root (manifest.parquet, etc).")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--resolution", choices=["360p", "720p"], default="360p")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--windows-per-round", type=int, default=1)
    parser.add_argument("--max-rounds-per-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--audio-encoder-name", default="audio_logmel",
                        help="Subdirectory under --out/audio_features used to namespace this feature set.")
    parser.add_argument("--labels-backend", choices=["auto", "match_states", "actions_bin"],
                        default="auto",
                        help="Source for action-probe labels. 'auto' tries match_states then "
                             "actions_bin. 'actions_bin' is the public-release path because the "
                             "canonical match_states aggregations are not redistributed.")
    parser.add_argument("--audio-probe-epochs", type=int, default=40)
    parser.add_argument("--audio-probe-batch-size", type=int, default=256)
    parser.add_argument("--audio-probe-lr", type=float, default=1e-3)
    parser.add_argument("--audio-probe-hidden-dim", type=int, default=256)
    parser.add_argument("--audio-probe-dropout", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--reuse-windows", action="store_true",
                        help="Skip window/label construction if output files already exist.")
    parser.add_argument("--reuse-features", action="store_true",
                        help="Skip audio feature extraction if features already exist.")
    add_wandb_args(parser)
    args = parser.parse_args()

    py = sys.executable
    shard_root = args.shard_root or args.root
    args.out.mkdir(parents=True, exist_ok=True)

    windows_path = args.out / "windows" / "eval_windows.parquet"
    labels_path = args.out / "action_probe" / "action_probe_labels.parquet"
    audio_features_dir = args.out / "audio_features" / args.audio_encoder_name
    probe_out = args.out / "audio_probe"

    if not (args.reuse_windows and windows_path.exists()):
        run([
            py, "-m", "cs2_release.core.windows",
            "--root", str(args.root),
            "--window-seconds", str(args.window_seconds),
            "--windows-per-round", str(args.windows_per_round),
            "--max-rounds-per-split", str(args.max_rounds_per_split),
            "--seed", str(args.seed),
            "--out", str(windows_path),
        ])
    if not (args.reuse_windows and labels_path.exists()):
        run([
            py, "-m", "cs2_release.action_probe.labels",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--backend", args.labels_backend,
            "--windows", str(windows_path),
            "--out", str(labels_path),
        ])

    audio_npz = audio_features_dir / "audio_features.npz"
    if not (args.reuse_features and audio_npz.exists()):
        run([
            py, "-m", "cs2_release.encoders.extract_audio",
            "--root", str(args.root),
            "--shard-root", str(shard_root),
            "--resolution", args.resolution,
            "--windows", str(windows_path),
            "--sample-rate", str(args.sample_rate),
            "--n-mels", str(args.n_mels),
            "--n-fft", str(args.n_fft),
            "--hop-length", str(args.hop_length),
            "--encoder-name", args.audio_encoder_name,
            "--out", str(args.out / "audio_features"),
        ])

    run([
        py, "-m", "cs2_release.audio_probe.train",
        "--labels", str(labels_path),
        "--features", str(audio_features_dir),
        "--epochs", str(args.audio_probe_epochs),
        "--batch-size", str(args.audio_probe_batch_size),
        "--lr", str(args.audio_probe_lr),
        "--hidden-dim", str(args.audio_probe_hidden_dim),
        "--dropout", str(args.audio_probe_dropout),
        "--device", args.device,
        "--seed", str(args.seed),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--out", str(probe_out),
    ] + wandb_args(args, stage="audio-probe"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
