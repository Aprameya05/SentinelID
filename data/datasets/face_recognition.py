"""
Dataset loaders for face recognition training.

Supported:
    MSCelebDataset     — MS-Celeb-1M (aligned, rec format or image folder)
    VGGFace2Dataset    — VGGFace2 (train/test split, image folder)
    CASIAWebFaceDataset — CASIA-WebFace (image folder)

All return (image_tensor, identity_label) where label is a contiguous
integer starting from 0 within each dataset. When combined with ConcatDataset,
use the offset mapping in FaceRecognitionDatasetBuilder to remap labels.
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .base import ImageFolderBase, default_transform


class MSCelebDataset(Dataset):
    """
    MS-Celeb-1M loader.

    Supports two formats:
      1. Image folder (post-alignment): root/identity_id/img.jpg
      2. MXNet .rec/.idx binary format (InsightFace style)

    Set `rec_format=True` for the binary format.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        image_size: int = 112,
        augment: bool = True,
        rec_format: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform or default_transform(image_size, augment)
        self.rec_format = rec_format
        self.samples: list[tuple[Path | int, int]] = []
        self.classes: list[str] = []

        if rec_format:
            self._load_rec()
        else:
            base = ImageFolderBase(root, self.transform, image_size, augment)
            self.samples = base.samples
            self.classes = base.classes
            self._base = base

    def _load_rec(self):
        """Parse InsightFace-style .rec + .idx file."""
        rec_path = self.root / "train.rec"
        idx_path = self.root / "train.idx"
        if not rec_path.exists() or not idx_path.exists():
            raise FileNotFoundError(f"MXNet .rec/.idx not found in {self.root}")

        try:
            import mxnet as mx
            self._record = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "r")
            # Read header to get number of samples
            header, _ = mx.recordio.unpack(self._record.read_idx(0))
            self._n_samples = int(header.label[0])
            self._start_idx = int(header.label[1])
        except ImportError:
            raise ImportError(
                "mxnet is required for .rec format. Install with: pip install mxnet"
            )

    def __len__(self) -> int:
        if self.rec_format:
            return self._n_samples
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.rec_format:
            import io

            import mxnet as mx
            header, s = mx.recordio.unpack(self._record.read_idx(idx + self._start_idx))
            label = int(header.label)
            img = Image.open(io.BytesIO(s)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self) -> int:
        if self.rec_format:
            return self._n_samples  # approximate; use actual class count from header
        return len(self.classes)


class VGGFace2Dataset(ImageFolderBase):
    """
    VGGFace2 dataset loader.

    Expected structure after download and alignment:
        root/
            train/
                n000001/  *.jpg
                n000002/  *.jpg
            test/
                n000001/  *.jpg

    Pass split='train' or split='test'.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 112,
        augment: bool = True,
    ):
        split_root = str(Path(root) / split)
        super().__init__(split_root, transform, image_size, augment)


class CASIAWebFaceDataset(ImageFolderBase):
    """
    CASIA-WebFace loader.

    Standard image folder structure:
        root/
            0000045/  *.jpg
            0000099/  *.jpg
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        image_size: int = 112,
        augment: bool = True,
    ):
        super().__init__(root, transform, image_size, augment)


class LFWPairsDataset(Dataset):
    """
    LFW pairs dataset for verification evaluation.

    Returns (img1, img2, label) where label=1 for genuine pair.
    Expects the standard lfw/pairs.txt format.
    """

    def __init__(self, root: str, image_size: int = 112):
        import torchvision.transforms as T
        self.root = Path(root)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.pairs: list[tuple[Path, Path, int]] = []
        self._parse_pairs()

    def _parse_pairs(self):
        pairs_file = self.root / "pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs.txt not found in {self.root}")

        with open(pairs_file) as f:
            lines = f.readlines()[1:]

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                name, n1, n2 = parts[0], int(parts[1]), int(parts[2])
                p1 = self.root / name / f"{name}_{n1:04d}.jpg"
                p2 = self.root / name / f"{name}_{n2:04d}.jpg"
                self.pairs.append((p1, p2, 1))
            elif len(parts) == 4:
                n1, n2 = parts[0], parts[2]
                p1 = self.root / n1 / f"{n1}_{int(parts[1]):04d}.jpg"
                p2 = self.root / n2 / f"{n2}_{int(parts[3]):04d}.jpg"
                self.pairs.append((p1, p2, 0))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p1, p2, label = self.pairs[idx]
        i1 = self.transform(Image.open(p1).convert("RGB"))
        i2 = self.transform(Image.open(p2).convert("RGB"))
        return i1, i2, torch.tensor(label, dtype=torch.float32)


class FaceRecognitionDatasetBuilder:
    """
    Utility to combine multiple face recognition datasets with remapped labels.

    Usage:
        builder = FaceRecognitionDatasetBuilder()
        builder.add("msceleb", MSCelebDataset(root1))
        builder.add("vggface2", VGGFace2Dataset(root2))
        combined, total_classes = builder.build()
    """

    def __init__(self):
        self._datasets: list[tuple[str, Dataset]] = []
        self._offsets: dict[str, int] = {}

    def add(self, name: str, dataset: Dataset):
        self._datasets.append((name, dataset))

    def build(self):
        from torch.utils.data import ConcatDataset

        offset = 0
        remapped = []
        for name, ds in self._datasets:
            remapped.append(_RemappedDataset(ds, offset))
            n = ds.num_classes if hasattr(ds, "num_classes") else len(set(s[1] for s in ds.samples))
            self._offsets[name] = offset
            offset += n

        return ConcatDataset(remapped), offset


class _RemappedDataset(Dataset):
    def __init__(self, dataset: Dataset, offset: int):
        self.dataset = dataset
        self.offset = offset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return img, label + self.offset
