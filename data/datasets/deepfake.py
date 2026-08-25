"""
Dataset loaders for deepfake detection training.

Supported:
    FaceForensicsDataset  — FaceForensics++ (4 manipulation types)
    CelebDFDataset        — CelebDF-v2
    WildDeepfakeDataset   — WildDeepfake (in-the-wild fakes)

All return (image_tensor, label) where label=0 for real, 1 for fake.
Face crops are assumed to be pre-extracted. Use scripts/preprocess_faces.py
to extract and align faces from raw videos.
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .base import default_transform

FF_MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
FF_COMPRESSIONS = ["c0", "c23", "c40"]  # raw, light, heavy


class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ dataset loader.

    Expected structure (after face extraction):
        root/
            original_sequences/
                youtube/
                    c23/faces/  <video_id>/  *.png
            manipulated_sequences/
                Deepfakes/
                    c23/faces/  <video_id>/  *.png
                Face2Face/ ...
                FaceShifter/ ...
                FaceSwap/ ...
                NeuralTextures/ ...

    Args:
        manipulation_types: subset of FF_MANIPULATION_TYPES to include
        compression: c0 (raw), c23 (light), c40 (heavy)
        max_frames_per_video: cap frames per video to avoid class imbalance within video
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        compression: str = "c23",
        manipulation_types: Optional[list[str]] = None,
        image_size: int = 224,
        augment: bool = True,
        max_frames_per_video: int = 300,
    ):
        self.transform = default_transform(image_size, augment)
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)

        # Load split file (train/val/test video IDs from FF++ repo)
        split_file = root / "splits" / f"{split}.json"
        if split_file.exists():
            import json
            with open(split_file) as f:
                split_ids = set(str(v) for pair in json.load(f) for v in pair)
        else:
            split_ids = None  # use all

        manip_types = manipulation_types or FF_MANIPULATION_TYPES

        # Real samples
        real_dir = root / "original_sequences" / "youtube" / compression / "faces"
        if real_dir.exists():
            for vid_dir in sorted(real_dir.iterdir()):
                if split_ids and vid_dir.name not in split_ids:
                    continue
                frames = sorted(vid_dir.glob("*.png"))[:max_frames_per_video]
                for p in frames:
                    self.samples.append((p, 0))

        # Fake samples
        for manip in manip_types:
            fake_dir = root / "manipulated_sequences" / manip / compression / "faces"
            if not fake_dir.exists():
                continue
            for vid_dir in sorted(fake_dir.iterdir()):
                if split_ids and vid_dir.name.split("_")[0] not in split_ids:
                    continue
                frames = sorted(vid_dir.glob("*.png"))[:max_frames_per_video]
                for p in frames:
                    self.samples.append((p, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(float(label))

    def class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        n_real = max(1, labels.count(0))
        n_fake = max(1, labels.count(1))
        return torch.tensor([1.0 / n_real if lbl == 0 else 1.0 / n_fake for lbl in labels])


class CelebDFDataset(Dataset):
    """
    CelebDF-v2 dataset loader.

    Expected structure (after face extraction):
        root/
            Celeb-real/      faces/  <video_id>/  *.jpg
            Celeb-synthesis/ faces/  <video_id>/  *.jpg
            YouTube-real/    faces/  <video_id>/  *.jpg

    Uses the official List_of_testing_videos.txt for test split.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        augment: bool = True,
        max_frames_per_video: int = 200,
    ):
        self.transform = default_transform(image_size, augment)
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)

        test_list_file = root / "List_of_testing_videos.txt"
        test_videos: set[str] = set()
        if test_list_file.exists():
            with open(test_list_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        test_videos.add(Path(parts[-1]).stem)

        def in_split(vid_name: str) -> bool:
            is_test = vid_name in test_videos
            return is_test if split == "test" else not is_test

        for cls_dir, label in [
            (root / "Celeb-real" / "faces", 0),
            (root / "YouTube-real" / "faces", 0),
            (root / "Celeb-synthesis" / "faces", 1),
        ]:
            if not cls_dir.exists():
                continue
            for vid_dir in sorted(cls_dir.iterdir()):
                if not in_split(vid_dir.name):
                    continue
                frames = sorted(vid_dir.glob("*.jpg"))[:max_frames_per_video]
                for p in frames:
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(float(label))

    def class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        n_real = max(1, labels.count(0))
        n_fake = max(1, labels.count(1))
        return torch.tensor([1.0 / n_real if lbl == 0 else 1.0 / n_fake for lbl in labels])


class WildDeepfakeDataset(Dataset):
    """
    WildDeepfake (in-the-wild) dataset.

    Structure:
        root/
            real_train/  seq_id/  *.jpg
            fake_train/  seq_id/  *.jpg
            real_test/   seq_id/  *.jpg
            fake_test/   seq_id/  *.jpg
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 224, augment: bool = True):
        self.transform = default_transform(image_size, augment)
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)
        suffix = "train" if split == "train" else "test"

        for label, cls in [(0, f"real_{suffix}"), (1, f"fake_{suffix}")]:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for seq in sorted(cls_dir.iterdir()):
                for p in sorted(seq.glob("*.jpg")):
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(float(label))
