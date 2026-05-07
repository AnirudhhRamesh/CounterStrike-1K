"""Pluggable frozen video/window encoders for release evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    dim: int
    resize: tuple[int, int]


class BaseEncoder:
    spec: EncoderSpec

    def encode(self, frames: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RGBHistEncoder(BaseEncoder):
    """Dependency-free visual baseline for smoke tests and offline reviewers."""

    def __init__(self, bins: int = 16, resize: tuple[int, int] = (64, 64)) -> None:
        self.bins = int(bins)
        name = "rgb_hist" if bins == 16 else f"rgb_hist_b{bins}"
        self.spec = EncoderSpec(name=name, dim=3 * bins + 6, resize=resize)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        arr = frames.astype(np.float32) / 255.0
        feats: list[np.ndarray] = []
        for channel in range(3):
            hist, _ = np.histogram(arr[..., channel], bins=self.bins, range=(0.0, 1.0), density=True)
            feats.append(hist.astype(np.float32))
        feats.append(arr.mean(axis=(0, 1, 2)).astype(np.float32))
        feats.append(arr.std(axis=(0, 1, 2)).astype(np.float32))
        out = np.concatenate(feats).astype(np.float32)
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out


class TorchvisionImageEncoder(BaseEncoder):
    """Average frame-level torchvision image features over a video window."""

    def __init__(
        self,
        *,
        model_name: str = "resnet18",
        device: str = "cuda",
        resize: tuple[int, int] = (224, 224),
    ) -> None:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.resize = resize
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            dim = int(model.fc.in_features)
            model.fc = nn.Identity()
            self.preprocess_mean = torch.tensor(weights.transforms().mean, device=self.device).view(1, 3, 1, 1)
            self.preprocess_std = torch.tensor(weights.transforms().std, device=self.device).view(1, 3, 1, 1)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            dim = int(model.fc.in_features)
            model.fc = nn.Identity()
            self.preprocess_mean = torch.tensor(weights.transforms().mean, device=self.device).view(1, 3, 1, 1)
            self.preprocess_std = torch.tensor(weights.transforms().std, device=self.device).view(1, 3, 1, 1)
        else:
            raise ValueError(f"unsupported torchvision image encoder: {model_name}")
        self.model = model.to(self.device).eval()
        self.spec = EncoderSpec(name=f"torchvision_{model_name}", dim=dim, resize=resize)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(frames).to(self.device).float().permute(0, 3, 1, 2) / 255.0
            x = (x - self.preprocess_mean) / self.preprocess_std
            feats = self.model(x)
            feat = feats.mean(dim=0)
            feat = torch.nn.functional.normalize(feat, dim=0)
            return feat.detach().cpu().numpy().astype(np.float32)


class DINOv2Encoder(BaseEncoder):
    """Average DINOv2 frame embeddings over a video window.

    Uses torch.hub so the dependency remains optional. Model weights are
    downloaded by PyTorch's normal cache on first use.
    """

    def __init__(
        self,
        *,
        variant: str = "dinov2_vits14",
        device: str = "cuda",
        resize: tuple[int, int] = (224, 224),
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", variant).to(self.device).eval()
        dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.spec = EncoderSpec(name=variant, dim=dims.get(variant, 0), resize=resize)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(frames).to(self.device).float().permute(0, 3, 1, 2) / 255.0
            x = (x - self.mean) / self.std
            feats = self.model(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    feats = feats["x_norm_clstoken"]
                elif "x_prenorm" in feats:
                    feats = feats["x_prenorm"]
                else:
                    raise ValueError(f"DINOv2 returned unsupported keys: {sorted(feats)}")
            feat = feats.mean(dim=0)
            feat = torch.nn.functional.normalize(feat, dim=0)
            return feat.detach().cpu().numpy().astype(np.float32)


def canonical_encoder_name(name: str) -> str:
    if name in {"rgb_hist", "rgb_hist_b16"}:
        return "rgb_hist"
    if name == "rgb_hist_b32":
        return "rgb_hist_b32"
    if name in {"torchvision_resnet18", "resnet18"}:
        return "torchvision_resnet18"
    if name in {"torchvision_resnet50", "resnet50"}:
        return "torchvision_resnet50"
    if name in {"dinov2", "dinov2_vits14"}:
        return "dinov2_vits14"
    if name == "dinov2_vitb14":
        return "dinov2_vitb14"
    return name


def build_encoder(name: str, *, device: str = "cuda") -> BaseEncoder:
    if name in {"rgb_hist", "rgb_hist_b16"}:
        return RGBHistEncoder(bins=16)
    if name == "rgb_hist_b32":
        return RGBHistEncoder(bins=32)
    if name in {"torchvision_resnet18", "resnet18"}:
        return TorchvisionImageEncoder(model_name="resnet18", device=device)
    if name in {"torchvision_resnet50", "resnet50"}:
        return TorchvisionImageEncoder(model_name="resnet50", device=device)
    if name in {"dinov2", "dinov2_vits14"}:
        return DINOv2Encoder(variant="dinov2_vits14", device=device)
    if name == "dinov2_vitb14":
        return DINOv2Encoder(variant="dinov2_vitb14", device=device)
    raise ValueError(
        f"unknown encoder {name!r}; expected rgb_hist, torchvision_resnet18, "
        "torchvision_resnet50, dinov2_vits14, or dinov2_vitb14"
    )
