"""Lightweight dataset reader for the dataset_viewer notebook.

Wraps a frozen manifest (produced by `cs2_train/scripts/freeze_manifest.py`)
or any equivalent JSON list of clip records, with on-demand fetch from S3 or
direct read from local disk.

The viewer notebook (dataset_viewer.ipynb) drives this; nothing else should
import from here. The training-time loader is `cs2_train.src.dataset.CSDataset`.

Usage:
    from cs2_train.baselines.diamond.dataset import DatasetViewer
    v = DatasetViewer.from_local("data/cs2-counterstrike-1k-100h", "manifest_100h.json")
    print(len(v), v.summary())
    sample = v[42]
    sample.video_path     # local mp4 (downloaded if needed)
    sample.actions_df     # pandas DataFrame
    sample.metadata       # dict
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


@dataclass
class Sample:
    index:        int
    clip:         dict           # raw manifest row
    video_path:   Path
    actions_path: Path
    metadata:     dict           # subset of clip useful for display

    @property
    def actions_df(self):
        import pandas as pd
        return pd.read_parquet(self.actions_path)

    def play_html(self, width: int = 640, controls: bool = True,
                  autoplay: bool = False, loop: bool = False) -> str:
        """Return an <video> tag pointing at the local mp4."""
        attrs = []
        if controls:  attrs.append("controls")
        if autoplay:  attrs.append("autoplay")
        if loop:      attrs.append("loop")
        return (f'<video {" ".join(attrs)} width="{width}" '
                f'src="{self.video_path}"></video>')


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


class DatasetViewer:
    """Index-based viewer over a manifest of CS2 clips.

    Two construction modes:

      DatasetViewer.from_local(root, manifest_name)
        For a local copy made by `cs2_train/src/download.py` or by manually
        rsyncing from S3. Looks up `<root>/<video_path>` and
        `<root>/<actions_path>` per the manifest's relative paths.

      DatasetViewer.from_s3(manifest_or_uri, cache_dir, region="us-east-1")
        Reads a manifest JSON (a local path OR an s3:// URI). Each call to
        `__getitem__` lazily downloads `video_s3` + `actions_s3` to
        `cache_dir`, and reuses the local copy on subsequent calls.

    The `debug_overlay` flag controls whether `__getitem__` returns the raw
    clip's mp4 or a freshly-rendered debug overlay video (with keyboard /
    mouse / weapon HUD). Default True. Re-renders the overlay only when the
    cached overlay is missing or older than the source clip.
    """

    def __init__(
        self,
        manifest: list[dict],
        *,
        local_root: Path | None = None,
        cache_dir: Path | None = None,
        region: str = "us-east-1",
        debug_overlay: bool = True,
    ):
        self.manifest = manifest
        self.local_root = Path(local_root) if local_root else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.region = region
        self.debug_overlay = debug_overlay
        self._s3 = None
        if self.cache_dir:
            (self.cache_dir / "videos").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "actions").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "debug").mkdir(parents=True, exist_ok=True)

    # ---- constructors ---------------------------------------------------------

    @classmethod
    def from_local(cls, root: str | Path, manifest_name: str = "manifest.json",
                   *, debug_overlay: bool = True) -> "DatasetViewer":
        root = Path(root)
        manifest = json.loads((root / manifest_name).read_text())
        return cls(manifest, local_root=root, debug_overlay=debug_overlay)

    @classmethod
    def from_manifest_file(cls, manifest_path: str | Path,
                           cache_dir: str | Path,
                           *, region: str = "us-east-1",
                           debug_overlay: bool = True) -> "DatasetViewer":
        """Read a frozen-manifest JSON; fetch clips lazily from the S3 URIs in it."""
        if isinstance(manifest_path, str) and manifest_path.startswith("s3://"):
            import boto3
            s3 = boto3.Session(region_name=region).client("s3")
            bucket, key = _parse_s3_uri(manifest_path)
            manifest = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        else:
            manifest = json.loads(Path(manifest_path).read_text())
        return cls(manifest, cache_dir=Path(cache_dir), region=region,
                   debug_overlay=debug_overlay)

    # ---- python container API -------------------------------------------------

    def __len__(self) -> int:
        return len(self.manifest)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(idx)
        clip = self.manifest[idx]

        if self.local_root is not None and "video_path" in clip:
            video_path = self.local_root / clip["video_path"]
            actions_path = self.local_root / clip["actions_path"]
            if not video_path.exists():
                raise FileNotFoundError(video_path)
        else:
            video_path = self._fetch("video", clip)
            actions_path = self._fetch("actions", clip)

        if self.debug_overlay:
            video_path = self._ensure_debug_overlay(clip, video_path, actions_path)

        meta = {
            "match_id":    clip.get("source_demo_id", clip.get("match_id", "")),
            "player_id":   str(clip.get("player_id", "")),
            "player_name": clip.get("player_name", ""),
            "clip_id":     int(clip.get("clip_id", 0)),
            "frames":      int(clip.get("frames", 0)),
            "duration_s":  float(clip.get("duration_s", 0.0)),
            "split":       clip.get("split", ""),
            "map":         clip.get("map", ""),
            "event":       clip.get("event", ""),
            "teams":       clip.get("teams", []),
        }
        return Sample(
            index=idx, clip=clip, video_path=video_path,
            actions_path=actions_path, metadata=meta,
        )

    # ---- helpers --------------------------------------------------------------

    def summary(self) -> dict:
        by_split: dict[str, int] = {}
        n_frames = 0
        for c in self.manifest:
            by_split.setdefault(c.get("split", "?"), 0)
            by_split[c.get("split", "?")] += 1
            n_frames += int(c.get("frames", 0))
        return {
            "n_clips":   len(self.manifest),
            "n_frames":  n_frames,
            "n_hours":   round(n_frames / 30.0 / 3600.0, 2),
            "by_split":  by_split,
            "n_matches": len({c.get("source_demo_id", c.get("match_id", "?"))
                              for c in self.manifest}),
            "n_players": len({str(c.get("player_id", "?")) for c in self.manifest}),
        }

    def _fetch(self, kind: str, clip: dict) -> Path:
        """Lazy fetch from S3 to cache_dir. kind ∈ {video, actions}."""
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is None — pass one to from_manifest_file()")
        if self._s3 is None:
            import boto3
            self._s3 = boto3.Session(region_name=self.region).client("s3")
        key = "video_s3" if kind == "video" else "actions_s3"
        ext = "mp4" if kind == "video" else "parquet"
        bucket, s3_key = _parse_s3_uri(clip[key])
        match_id = clip.get("source_demo_id", clip.get("match_id", "x"))
        player_id = clip.get("player_id", "x")
        clip_id = int(clip.get("clip_id", 0))
        local = (self.cache_dir / kind / match_id /
                 f"player_{player_id}__clip_{clip_id:03d}.{ext}")
        if local.exists() and local.stat().st_size > 0:
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket, s3_key, str(local))
        return local

    def _ensure_debug_overlay(self, clip: dict, video_path: Path,
                              actions_path: Path) -> Path:
        """Re-render the debug overlay if missing or stale."""
        if self.local_root is not None:
            overlay_dir = self.local_root / "debug_views"
        elif self.cache_dir is not None:
            overlay_dir = self.cache_dir / "debug"
        else:
            return video_path
        match_id = clip.get("source_demo_id", clip.get("match_id", "x"))
        player_id = clip.get("player_id", "x")
        clip_id = int(clip.get("clip_id", 0))
        out_path = overlay_dir / match_id / f"player_{player_id}__clip_{clip_id:03d}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if (out_path.exists() and out_path.stat().st_size > 0
                and out_path.stat().st_mtime > video_path.stat().st_mtime):
            return out_path

        try:
            repo_root = Path(__file__).resolve().parents[3]
            cs2_generate = repo_root / "cs2_generate"
            if str(cs2_generate) not in sys.path:
                sys.path.insert(0, str(cs2_generate))
            from cs2_video_renderer import CS2VideoRenderer
            import pandas as pd
            renderer = CS2VideoRenderer()
            actions_df = pd.read_parquet(actions_path)
            renderer.render_video(
                str(video_path), actions_df,
                fps=float(clip.get("fps", 30)),
                output_path=str(out_path),
                header_info={
                    "player_name": clip.get("player_name", ""),
                    "player_id":   str(clip.get("player_id", "")),
                    "max_tick":    int(clip.get("end_tick", 0)),
                },
            )
        except Exception as e:  # noqa: BLE001
            print(f"  debug overlay render failed: {e!r}; falling back to raw mp4")
            return video_path
        return out_path
