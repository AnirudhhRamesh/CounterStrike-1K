"""Private S3 publisher for live DIAMOND validation and rollout artifacts."""

from __future__ import annotations

import json
import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from botocore.exceptions import ClientError


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3://bucket/prefix, got {uri!r}")
    return parsed.netloc, parsed.path.strip("/")


class ReviewPublisher:
    """Upload artifacts and maintain one pollable, pre-signed review index."""

    def __init__(
        self,
        *,
        s3_uri: str,
        run_id: str,
        arm: str,
        local_dir: Path,
        expires_seconds: int = 604_800,
    ) -> None:
        import boto3

        if expires_seconds < 60 or expires_seconds > 604_800:
            raise ValueError("S3 pre-signed URL expiry must be in [60, 604800] seconds")
        self.bucket, self.prefix = parse_s3_uri(s3_uri)
        self.run_id = run_id
        self.arm = arm
        self.local_dir = local_dir
        self.expires_seconds = expires_seconds
        self.client = boto3.client("s3")
        self.index_key = "/".join(
            part for part in (self.prefix, run_id, "index.json") if part
        )
        self.local_index = local_dir / "review_index.json"

    def _read_index(self) -> dict:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.index_key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {
                "NoSuchKey",
                "404",
            }:
                raise
            return {
                "schema_version": 1,
                "run_id": self.run_id,
                "updated_at_utc": None,
                "arms": {},
            }
        return json.loads(response["Body"].read())

    def _presign(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.expires_seconds,
        )

    def _upload(self, path: Path, key: str) -> dict:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.client.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "ServerSideEncryption": "AES256",
            },
        )
        return {
            "key": key,
            "url": self._presign(key),
            "size_bytes": path.stat().st_size,
        }

    def publish(
        self,
        *,
        step: int,
        kind: str,
        artifacts: dict[str, Path],
        metrics: dict,
    ) -> str:
        """Publish one eval group and return the pollable index URL."""

        uploaded = {}
        for label, path in artifacts.items():
            key = "/".join(
                part
                for part in (
                    self.prefix,
                    self.run_id,
                    self.arm,
                    f"step_{step:07d}",
                    path.name,
                )
                if part
            )
            uploaded[label] = self._upload(path, key)

        index = self._read_index()
        arm_payload = index.setdefault("arms", {}).setdefault(
            self.arm,
            {"status": "running", "events": []},
        )
        arm_payload["status"] = "running"
        event = {
            "step": int(step),
            "kind": kind,
            "published_at_utc": datetime.now(UTC).isoformat(),
            "artifacts": uploaded,
            "metrics": metrics,
        }
        events = [
            prior
            for prior in arm_payload["events"]
            if not (prior.get("step") == step and prior.get("kind") == kind)
        ]
        events.append(event)
        arm_payload["events"] = sorted(
            events,
            key=lambda item: (int(item["step"]), str(item["kind"])),
        )
        index["updated_at_utc"] = datetime.now(UTC).isoformat()
        encoded = json.dumps(index, indent=2).encode()
        self.local_index.write_bytes(encoded)
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.index_key,
            Body=encoded,
            ContentType="application/json",
            ServerSideEncryption="AES256",
            CacheControl="no-store",
        )
        index_url = self._presign(self.index_key)
        (self.local_dir / "review_index_url.txt").write_text(
            index_url + "\n",
            encoding="utf-8",
        )
        return index_url

    def mark_complete(self, *, final_step: int) -> str:
        index = self._read_index()
        arm_payload = index.setdefault("arms", {}).setdefault(
            self.arm,
            {"events": []},
        )
        arm_payload["status"] = "complete"
        arm_payload["final_step"] = int(final_step)
        arm_payload["completed_at_utc"] = datetime.now(UTC).isoformat()
        index["updated_at_utc"] = datetime.now(UTC).isoformat()
        encoded = json.dumps(index, indent=2).encode()
        self.local_index.write_bytes(encoded)
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.index_key,
            Body=encoded,
            ContentType="application/json",
            ServerSideEncryption="AES256",
            CacheControl="no-store",
        )
        return self._presign(self.index_key)
