"""Re-sign a raw private S3 review index with stable reviewer credentials.

GPU instances commonly use temporary role credentials. Their pre-signed URLs
expire when the role session rotates, even when a longer URL lifetime was
requested. This publisher therefore runs under non-session reviewer
credentials, replaces every artifact URL from its private object key, and
atomically overwrites a stable viewer-index key that the owner-only site polls.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3


def assert_private_bucket(client, bucket: str) -> None:
    block = client.get_public_access_block(Bucket=bucket)[
        "PublicAccessBlockConfiguration"
    ]
    required = (
        "BlockPublicAcls",
        "IgnorePublicAcls",
        "BlockPublicPolicy",
        "RestrictPublicBuckets",
    )
    if not all(block.get(key) is True for key in required):
        raise ValueError(f"{bucket} does not have all four S3 public-access blocks")


def signed_view(
    client,
    *,
    bucket: str,
    raw: dict,
    expires_seconds: int,
) -> dict:
    viewer = copy.deepcopy(raw)
    for arm in viewer.get("arms", {}).values():
        for event in arm.get("events", []):
            for artifact in event.get("artifacts", {}).values():
                key = artifact["key"]
                artifact["url"] = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expires_seconds,
                )
    viewer["viewer_index_generated_at_utc"] = datetime.now(UTC).isoformat()
    viewer["viewer_url_expires_seconds"] = expires_seconds
    return viewer


def publish_once(
    client,
    *,
    bucket: str,
    raw_key: str,
    viewer_key: str,
    expires_seconds: int,
) -> str:
    response = client.get_object(Bucket=bucket, Key=raw_key)
    raw = json.loads(response["Body"].read())
    viewer = signed_view(
        client,
        bucket=bucket,
        raw=raw,
        expires_seconds=expires_seconds,
    )
    client.put_object(
        Bucket=bucket,
        Key=viewer_key,
        Body=json.dumps(viewer, indent=2).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
        CacheControl="no-store",
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": viewer_key},
        ExpiresIn=expires_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--raw-key", required=True)
    parser.add_argument("--viewer-key", required=True)
    parser.add_argument("--expires-seconds", type=int, default=604_800)
    parser.add_argument("--url-output", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--allow-temporary-credentials", action="store_true")
    args = parser.parse_args()

    if not 60 <= args.expires_seconds <= 604_800:
        raise ValueError("S3 pre-signed expiry must be in [60, 604800]")
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise ValueError("no AWS credentials are available")
    if (
        credentials.get_frozen_credentials().token is not None
        and not args.allow_temporary_credentials
    ):
        raise ValueError(
            "refusing temporary AWS credentials: viewer URLs would expire with the session"
        )

    client = boto3.client("s3")
    assert_private_bucket(client, args.bucket)
    while True:
        try:
            viewer_url = publish_once(
                client,
                bucket=args.bucket,
                raw_key=args.raw_key,
                viewer_key=args.viewer_key,
                expires_seconds=args.expires_seconds,
            )
            args.url_output.parent.mkdir(parents=True, exist_ok=True)
            args.url_output.write_text(viewer_url + "\n", encoding="utf-8")
            print(
                f"{datetime.now(UTC).isoformat()} published signed viewer index",
                flush=True,
            )
        except Exception as exc:
            print(
                f"{datetime.now(UTC).isoformat()} publish failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if not args.watch:
                raise
        if not args.watch:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
