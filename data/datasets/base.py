"""
Base dataset class shared across all SentinelID dataset loaders.
"""

from pathlib import Path
from typing import Callable, Optional

import torchvision.transforms as T
from torch.utils.data import Dataset


def default_transform(image_size: int = 224, augment: bool = False) -> T.Compose:
    ops = [T.Resize((image_size, image_size))]
    if augment:
        ops += [
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.RandomErasing(p=0.2),
        ]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)


class ImageFolderBase(Dataset):
    """
    Generic image folder dataset.
    Expected structure:
        root/
            class_a/  img1.jpg  img2.jpg ...
            class_b/  ...
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        image_size: int = 112,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform or default_transform(image_size, augment)
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self._scan()

    def _scan(self):
        class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        self.classes = [d.name for d in class_dirs]
        for label, d in enumerate(class_dirs):
            for ext in self.EXTENSIONS:
                for p in d.glob(f"*{ext}"):
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)
