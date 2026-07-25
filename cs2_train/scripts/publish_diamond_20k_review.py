"""Publish the resource-amended DIAMOND step-20k private review endpoint."""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import UTC, datetime
from pathlib import Path

import boto3

from .publish_signed_review_index import assert_private_bucket


def upload_artifact(client, *, bucket: str, key: str, path: Path) -> dict:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type, "ServerSideEncryption": "AES256"},
    )
    return {"key": key, "size_bytes": path.stat().st_size}


def update_index(
    raw: dict,
    *,
    window: str,
    event: dict,
    step: int,
) -> None:
    true_arm = raw.setdefault("arms", {}).setdefault(
        "true",
        {"events": [], "status": "running"},
    )
    kind = f"confirmatory_{window}"
    true_arm["events"] = [
        prior
        for prior in true_arm.get("events", [])
        if not (int(prior.get("step", -1)) == step and prior.get("kind") == kind)
    ]
    true_arm["events"].append(event)
    true_arm["events"] = sorted(
        true_arm["events"],
        key=lambda item: (int(item["step"]), str(item["kind"])),
    )
    true_arm["status"] = "complete"
    true_arm["final_step"] = step
    raw["arms"]["shuffled"] = {
        "events": [],
        "status": "skipped_resource_amendment",
        "reason": (
            "The separately trained shuffled arm was removed before test access "
            "to meet the rebuttal deadline."
        ),
    }
    raw["final_results_status"] = "complete_resource_amended"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, default=20_000)
    parser.add_argument("--videos-per-window", type=int, default=4)
    args = parser.parse_args()

    if not (args.endpoint_root / "COMPLETE").is_file():
        raise FileNotFoundError("endpoint completion marker is absent")
    client = boto3.client("s3")
    assert_private_bucket(client, args.bucket)
    raw_key = "/".join((args.prefix.strip("/"), args.run_id, "index.json"))
    raw = json.loads(client.get_object(Bucket=args.bucket, Key=raw_key)["Body"].read())
    amendment = args.endpoint_root / "provenance" / "resource_amendment.json"

    for window in ("midpoint", "first-death"):
        evaluation = args.endpoint_root / window
        summary_path = evaluation / "summary.json"
        motion_path = evaluation / "motion_metrics" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        motion = json.loads(motion_path.read_text(encoding="utf-8"))
        review_manifest = json.loads(
            (evaluation / "review_manifest.json").read_text(encoding="utf-8")
        )
        local_artifacts = {
            "evaluation_summary": summary_path,
            "motion_summary": motion_path,
            "resource_amendment": amendment,
        }
        for index, item in enumerate(review_manifest[: args.videos_per_window]):
            local_artifacts[f"paired_trace_{index:02d}"] = evaluation / item["path"]

        uploaded = {}
        for label, path in local_artifacts.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            key = "/".join(
                (
                    args.prefix.strip("/"),
                    args.run_id,
                    "true",
                    "confirmatory",
                    window,
                    "step-20000",
                    f"{label}-{path.name}",
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
            "kind": f"confirmatory_{window}",
            "published_at_utc": datetime.now(UTC).isoformat(),
            "artifacts": uploaded,
            "metrics": {
                "resource_amended": True,
                "num_eval_samples": summary["num_eval_samples"],
                "num_rounds": summary["num_rounds"],
                "eval_seeds": summary["eval_seeds"],
                "columns": ["ground_truth", *summary["action_modes"]],
                "means": summary["means"],
                "paired_deltas": summary["paired_deltas"],
                "motion_means": motion["means"],
                "motion_paired_deltas": motion["paired_deltas"],
            },
        }
        update_index(
            raw,
            window=window,
            event=event,
            step=args.step,
        )

    raw["updated_at_utc"] = datetime.now(UTC).isoformat()
    client.put_object(
        Bucket=args.bucket,
        Key=raw_key,
        Body=json.dumps(raw, indent=2).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
        CacheControl="no-store",
    )
    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "raw_index_key": raw_key,
                "status": raw["final_results_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
