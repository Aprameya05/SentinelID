"""
Dataset loader for document intelligence training.

Supported:
    MIDV500Dataset  — MIDV-500 (50 document types x 10 clips each)
    MIDV2020Dataset — MIDV-2020 (updated version, 10 document types)

Both datasets contain identity document images with:
    - Document type labels
    - Field annotations (name, DOB, doc number, expiry)
    - No forgery labels (genuine documents only; synthetic forgeries added via augmentation)

Forgery augmentation (applied at training time):
    - Random text region copy-paste
    - Font substitution via PIL
    - JPEG re-compression artifacts
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

DOC_TYPE_MAP = {
    "passport": 0,
    "id_card": 1,
    "drivers_license": 2,
    "residence_permit": 3,
    "visa": 4,
    "other": 5,
}


def _normalize_bbox(bbox: list[float], W: int, H: int) -> list[int]:
    """Convert [x, y, w, h] to LayoutLM [x0, y0, x1, y1] in [0, 1000]."""
    x, y, w, h = bbox
    x0 = int(x / W * 1000)
    y0 = int(y / H * 1000)
    x1 = int((x + w) / W * 1000)
    y1 = int((y + h) / H * 1000)
    return [
        max(0, min(1000, x0)),
        max(0, min(1000, y0)),
        max(0, min(1000, x1)),
        max(0, min(1000, y1)),
    ]


class MIDV500Dataset(Dataset):
    """
    MIDV-500 dataset loader.

    Expected structure after download:
        root/
            <doc_type>/
                clips/  <clip_id>/  <frame_id>/
                    image.tif
                    ground_truth.json

    ground_truth.json contains field annotations and bounding boxes.

    Returns:
        image: (3, 224, 224) tensor
        token_ids: (L,) int64 from a pre-tokenized annotation (or zeros if unavailable)
        bbox: (L, 4) int64 in [0, 1000]
        attention_mask: (L,) int64
        doc_type_label: int
        forgery_label: float (0.0 = genuine; 1.0 = augmented forgery)
        field_labels: dict — not returned here, use collate_fn
    """

    MAX_SEQ_LEN = 128
    IMAGE_SIZE = 224

    def __init__(
        self,
        root: str,
        augment: bool = True,
        forgery_aug_prob: float = 0.3,
        tokenizer=None,
    ):
        self.root = Path(root)
        self.augment = augment
        self.forgery_aug_prob = forgery_aug_prob
        self.tokenizer = tokenizer
        self.samples: list[dict] = []
        self.image_transform = T.Compose([
            T.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._scan()

    def _scan(self):
        for doc_dir in sorted(self.root.iterdir()):
            if not doc_dir.is_dir():
                continue
            doc_type = doc_dir.name.lower()
            clips_dir = doc_dir / "clips"
            if not clips_dir.exists():
                continue
            for clip in sorted(clips_dir.iterdir()):
                for frame in sorted(clip.iterdir()):
                    img_path = frame / "image.tif"
                    gt_path = frame / "ground_truth.json"
                    if img_path.exists():
                        self.samples.append({
                            "image_path": img_path,
                            "gt_path": gt_path if gt_path.exists() else None,
                            "doc_type": doc_type,
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_annotations(self, gt_path: Path, W: int, H: int) -> tuple[list[str], list[list[int]]]:
        """Load OCR word tokens and their bounding boxes from ground_truth.json."""
        try:
            with open(gt_path) as f:
                gt = json.load(f)
            words, bboxes = [], []
            for field in gt.get("fields", []):
                text = str(field.get("value", ""))
                bbox_raw = field.get("bbox", [0, 0, 10, 10])
                words.append(text)
                bboxes.append(_normalize_bbox(bbox_raw, W, H))
            return words, bboxes
        except Exception:
            return [], []

    def _forgery_augment(self, image: Image.Image) -> tuple[Image.Image, float]:
        """Apply a random forgery-like augmentation. Returns (modified_image, forgery_label)."""
        aug_type = random.choice(["copypaste", "jpeg_blast", "color_shift"])
        img = image.copy()
        W, H = img.size

        if aug_type == "copypaste":
            # Copy a random region and paste it elsewhere
            x1 = random.randint(0, W // 2)
            y1 = random.randint(0, H // 2)
            x2 = x1 + random.randint(20, W // 4)
            y2 = y1 + random.randint(10, H // 4)
            region = img.crop((x1, y1, x2, y2))
            px = random.randint(0, W - (x2 - x1))
            py = random.randint(0, H - (y2 - y1))
            img.paste(region, (px, py))

        elif aug_type == "jpeg_blast":
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(10, 30))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        elif aug_type == "color_shift":
            arr = np.array(img, dtype=np.float32)
            ch = random.randint(0, 2)
            arr[:, :, ch] = np.clip(arr[:, :, ch] * (0.5 + random.random()), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))

        return img, 1.0

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        W, H = image.size

        # Forgery augmentation
        forgery_label = 0.0
        if self.augment and random.random() < self.forgery_aug_prob:
            image, forgery_label = self._forgery_augment(image)

        # Image tensor
        image_t = self.image_transform(image)

        # Annotations
        words, bboxes = [], []
        if sample["gt_path"]:
            words, bboxes = self._load_annotations(sample["gt_path"], W, H)

        L = self.MAX_SEQ_LEN
        # Tokenize
        if self.tokenizer and words:
            enc = self.tokenizer(
                words,
                is_split_into_words=True,
                max_length=L,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            token_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            # Align bboxes to token positions (repeat bbox for each sub-token)
            word_ids = enc.word_ids()
            bbox_per_token = []
            for wid in word_ids:
                if wid is None:
                    bbox_per_token.append([0, 0, 0, 0])
                elif wid < len(bboxes):
                    bbox_per_token.append(bboxes[wid])
                else:
                    bbox_per_token.append([0, 0, 0, 0])
            bbox_t = torch.tensor(bbox_per_token, dtype=torch.long)
        else:
            token_ids = torch.zeros(L, dtype=torch.long)
            attention_mask = torch.zeros(L, dtype=torch.long)
            bbox_t = torch.zeros(L, 4, dtype=torch.long)

        doc_type_str = sample["doc_type"]
        for key in DOC_TYPE_MAP:
            if key in doc_type_str:
                doc_type_label = DOC_TYPE_MAP[key]
                break
        else:
            doc_type_label = DOC_TYPE_MAP["other"]

        return {
            "image": image_t,
            "token_ids": token_ids,
            "bbox": bbox_t,
            "attention_mask": attention_mask,
            "doc_type_label": torch.tensor(doc_type_label, dtype=torch.long),
            "forgery_label": torch.tensor(forgery_label, dtype=torch.float32),
        }


class MIDV2020Dataset(MIDV500Dataset):
    """
    MIDV-2020 — same structure as MIDV-500 with 10 updated document types.
    Inherits all functionality; just points to a different root.
    """
    pass
