from __future__ import annotations

from cs2_train.scripts.publish_signed_review_index import signed_view


class _Client:
    def generate_presigned_url(
        self, _operation: str, *, Params: dict, ExpiresIn: int
    ) -> str:
        return f"signed://{Params['Bucket']}/{Params['Key']}?expires={ExpiresIn}"


def test_signed_view_replaces_temporary_gpu_urls_without_mutating_raw() -> None:
    raw = {
        "arms": {
            "true": {
                "events": [
                    {
                        "artifacts": {
                            "grid": {
                                "key": "private/grid.png",
                                "url": "temporary-instance-role-url",
                            }
                        }
                    }
                ]
            }
        }
    }
    viewer = signed_view(
        _Client(),
        bucket="bucket",
        raw=raw,
        expires_seconds=604_800,
    )
    assert (
        viewer["arms"]["true"]["events"][0]["artifacts"]["grid"]["url"]
        == "signed://bucket/private/grid.png?expires=604800"
    )
    assert (
        raw["arms"]["true"]["events"][0]["artifacts"]["grid"]["url"]
        == "temporary-instance-role-url"
    )
