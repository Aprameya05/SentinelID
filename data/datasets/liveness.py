"""
Dataset loaders for liveness and anti-spoofing training.

Supported:
    NUAADataset       — NUAA Imposter Database (print attacks)
    CASIASURFDataset  — CASIA-SURF (RGB + Depth + IR)
    SiWDataset        — Spoof in Wild (diverse spoof types)
    MSUMFSDDataset    — MSU-MFSD (replay + print attacks, video frames)

All return (image_tensor, depth_tensor, liveness_label, has_depth).
depth_tensor is zero-filled when ground-truth depth is unavailable.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .base import default_transform


class NUAADataset(Dataset):
    """
    NUAA Imposter Database.

    Expected structure (after extraction):
        root/
            ClientFace/      <- genuine (live)
                0001/  *.jpg
            ImposterFace/    <- attack (spoof)
                0001/  *.jpg

    Returns (image, zeros_depth, label, has_depth=0).
    """

    def __init__(self, root: str, image_size: int = 256, augment: bool = True):
        self.transform = default_transform(image_size, augment)
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)

        live_dir = root / "ClientFace"
        spoof_dir = root / "ImposterFace"
        self.image_size = image_size

        for subdir in sorted(live_dir.glob("*")) if live_dir.exists() else []:
            for p in subdir.glob("*.jpg"):
                self.samples.append((p, 1))

        for subdir in sorted(spoof_dir.glob("*")) if spoof_dir.exists() else []:
            for p in subdir.glob("*.jpg"):
                self.samples.append((p, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        depth = torch.zeros(1, self.image_size // 4, self.image_size // 4)
        return img_t, depth, torch.tensor(float(label)), torch.tensor(0.0)


class CASIASURFDataset(Dataset):
    """
    CASIA-SURF dataset (RGB + Depth + IR multimodal).

    Expected structure:
        root/
            train/
                real/   <profile_id>/color/*.jpg  depth/*.jpg  ir/*.jpg
                fake/   <profile_id>/...
            val/

    When depth maps are available, returns them as supervision targets.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        use_depth: bool = True,
    ):
        self.transform = default_transform(image_size, augment)
        self.image_size = image_size
        self.use_depth = use_depth
        self.samples: list[tuple[Path, Optional[Path], int]] = []

        root_split = Path(root) / split
        for label, cls in [(1, "real"), (0, "fake")]:
            cls_dir = root_split / cls
            if not cls_dir.exists():
                continue
            for profile in sorted(cls_dir.iterdir()):
                color_dir = profile / "color"
                depth_dir = profile / "depth"
                if not color_dir.exists():
                    continue
                for color_img in sorted(color_dir.glob("*.jpg")):
                    depth_img = depth_dir / color_img.name if depth_dir.exists() else None
                    if depth_img and not depth_img.exists():
                        depth_img = None
                    self.samples.append((color_img, depth_img, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        color_path, depth_path, label = self.samples[idx]

        img = Image.open(color_path).convert("RGB")
        img_t = self.transform(img)

        h = w = self.image_size // 4
        if depth_path and self.use_depth:
            depth_img = Image.open(depth_path).convert("L")
            depth_arr = np.array(depth_img, dtype=np.float32) / 255.0
            depth_t = torch.from_numpy(depth_arr).unsqueeze(0)
            depth_t = torch.nn.functional.interpolate(
                depth_t.unsqueeze(0), size=(h, w)
            ).squeeze(0)
            has_depth = torch.tensor(1.0)
        else:
            depth_t = torch.zeros(1, h, w)
            has_depth = torch.tensor(0.0)

        return img_t, depth_t, torch.tensor(float(label)), has_depth


class SiWDataset(Dataset):
    """
    Spoof in Wild (SiW) dataset.

    Structure:
        root/
            train/
                live/   subject_id/session_id/frame_*.jpg
                spoof/  subject_id/spoof_type/frame_*.jpg
            test/
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 256, augment: bool = True):
        self.transform = default_transform(image_size, augment)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []

        root_split = Path(root) / split
        for label, cls in [(1, "live"), (0, "spoof")]:
            cls_dir = root_split / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*.jpg"):
                self.samples.append((p, label))
            for p in cls_dir.rglob("*.png"):
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        depth = torch.zeros(1, self.image_size // 4, self.image_size // 4)
        return img_t, depth, torch.tensor(float(label)), torch.tensor(0.0)


class MSUMFSDDataset(Dataset):
    """
    MSU Mobile Face Spoofing Database.

    Structure:
        root/
            real/   subject_id/  *.jpg (extracted frames)
            attack/ subject_id/  *.jpg

    attack/ contains both print and replay attacks mixed; no sub-type split needed
    for binary liveness training.
    """

    def __init__(self, root: str, image_size: int = 256, augment: bool = True):
        self.transform = default_transform(image_size, augment)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []

        root = Path(root)
        for label, cls in [(1, "real"), (0, "attack")]:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*.jpg"):
                self.samples.append((p, label))
            for p in cls_dir.rglob("*.png"):
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        depth = torch.zeros(1, self.image_size // 4, self.image_size // 4)
        return img_t, depth, torch.tensor(float(label)), torch.tensor(0.0)
