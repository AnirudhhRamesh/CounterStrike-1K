"""Optional Weights & Biases tracking helpers for release evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def add_wandb_args(parser) -> None:
    parser.add_argument("--wandb-project", default=None,
                        help="Enable W&B logging to this project.")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None,
                        help="Comma-separated tags for W&B.")
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-artifacts", action="store_true",
                        help="Upload output metrics/tables/checkpoints as W&B artifacts.")


def init_wandb(args, *, job_type: str, config: dict[str, Any] | None = None):
    if getattr(args, "wandb_mode", "online") == "disabled":
        return None
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required when --wandb-project is set") from exc
    tags_arg = getattr(args, "wandb_tags", None)
    tags = [tag.strip() for tag in tags_arg.split(",") if tag.strip()] if tags_arg else None
    return wandb.init(
        project=project,
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        group=getattr(args, "wandb_group", None),
        job_type=job_type,
        mode=getattr(args, "wandb_mode", "online"),
        tags=tags,
        config=config,
    )


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def flatten_metrics(payload: dict[str, Any], *, prefix: str = "") -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_metrics(value, prefix=name))
        elif isinstance(value, (str, int, float, bool)):
            out[name] = value
        elif isinstance(value, np.generic):
            out[name] = value.item()
    return out


def log_metrics(
    run,
    metrics: dict[str, Any],
    *,
    prefix: str = "",
    step: int | None = None,
    summary: bool = False,
) -> None:
    if run is None:
        return
    flat = flatten_metrics(metrics, prefix=prefix)
    if flat:
        run.log(flat, step=step)
    if summary:
        for key, value in flat.items():
            run.summary[key] = value


def log_dataframe(run, key: str, df: pd.DataFrame, *, max_rows: int = 200) -> None:
    if run is None or df.empty:
        return
    import wandb

    table_df = df.head(max_rows).copy()
    run.log({key: wandb.Table(dataframe=table_df)})


def log_images(run, key: str, images: list[tuple[np.ndarray, str]]) -> None:
    if run is None or not images:
        return
    import wandb

    run.log({key: [wandb.Image(image, caption=caption) for image, caption in images]})


def log_artifact(run, *, name: str, artifact_type: str, paths: list[str | Path]) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type)
    added = False
    for path_like in paths:
        path = Path(path_like)
        if path.exists():
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            added = True
    if added:
        run.log_artifact(artifact)
