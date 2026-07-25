"""Publish sanitized final DIAMOND evaluator artifacts to the private review index."""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import UTC, datetime
from pathlib import Path

import boto3

from .publish_signed_review_index import assert_private_bucket

LOCAL_PATH_FIELDS = {
    "checkpoint",
    "config",
    "manifest",
    "sample_plan",
    "review_manifest",
    "per_sample_metrics",
}


def upload_artifact(client, *, bucket: str, key: str, path: Path) -> dict:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": content_type,
            "ServerSideEncryption": "AES256",
        },
    )
    return {"key": key, "size_bytes": path.stat().st_size}


def write_sanitized_summary(summary: dict, path: Path) -> Path:
    sanitized = {
        key: value for key, value in summary.items() if key not in LOCAL_PATH_FIELDS
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitized, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, default=50_000)
    parser.add_argument("--videos-per-eval", type=int, default=4)
    args = parser.parse_args()

    client = boto3.client("s3")
    assert_private_bucket(client, args.bucket)
    raw_key = "/".join((args.prefix.strip("/"), args.run_id, "index.json"))
    raw = json.loads(client.get_object(Bucket=args.bucket, Key=raw_key)["Body"].read())
    combined_paths = (
        args.run_root / "evaluation" / "rebuttal_summary.json",
        args.run_root / "evaluation" / "rebuttal_summary.md",
    )
    if not all(path.is_file() for path in combined_paths):
        raise FileNotFoundError("run summarize_diamond_rebuttal.py before final publishing")

    for arm in ("true", "shuffled"):
        arm_payload = raw["arms"][arm]
        for window_mode in ("midpoint", "first-death"):
            evaluation = args.run_root / arm / "evaluation" / window_mode
            summary_path = evaluation / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            review_manifest = json.loads(
                (evaluation / "review_manifest.json").read_text(encoding="utf-8")
            )
            public_summary_path = write_sanitized_summary(
                summary,
                args.run_root
                / "evaluation"
                / "publish"
                / f"{arm}-{window_mode}-summary.json",
            )
            local_artifacts = {
                "evaluation_summary_json": public_summary_path,
                "cross_arm_summary_json": combined_paths[0],
                "cross_arm_summary_markdown": combined_paths[1],
            }
            for index, item in enumerate(review_manifest[: args.videos_per_eval]):
                local_artifacts[f"paired_trace_{index:02d}"] = evaluation / item["path"]

            uploaded = {}
            for label, path in local_artifacts.items():
                if not path.is_file():
                    raise FileNotFoundError(path)
                key = "/".join(
                    (
                        args.prefix.strip("/"),
                        args.run_id,
                        arm,
                        "confirmatory",
                        window_mode,
                        path.name,
                    )
                )
                uploaded[label] = upload_artifact(
                    client,
                    bucket=args.bucket,
                    key=key,
                    path=path,
                )

            event = {
                "step": args.step,
                "kind": f"confirmatory_{window_mode}",
                "published_at_utc": datetime.now(UTC).isoformat(),
                "artifacts": uploaded,
                "metrics": {
                    "num_eval_samples": summary["num_eval_samples"],
                    "num_rounds": summary["num_rounds"],
                    "eval_seeds": summary["eval_seeds"],
                    "columns": ["ground_truth", *summary["action_modes"]],
                    "means": summary["means"],
                    "paired_deltas": summary["paired_deltas"],
                },
            }
            arm_payload["events"] = [
                prior
                for prior in arm_payload.get("events", [])
                if not (
                    int(prior.get("step", -1)) == args.step
                    and prior.get("kind") == event["kind"]
                )
            ]
            arm_payload["events"].append(event)
            arm_payload["events"] = sorted(
                arm_payload["events"],
                key=lambda item: (int(item["step"]), str(item["kind"])),
            )
        arm_payload["status"] = "complete"
        arm_payload["final_step"] = args.step

    raw["updated_at_utc"] = datetime.now(UTC).isoformat()
    raw["experiment"]["status"] = "complete"
    raw["final_results_status"] = "complete"
    encoded = json.dumps(raw, indent=2).encode()
    client.put_object(
        Bucket=args.bucket,
        Key=raw_key,
        Body=encoded,
        ContentType="application/json",
        ServerSideEncryption="AES256",
        CacheControl="no-store",
    )
    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "raw_index_key": raw_key,
                "status": "complete",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
