"""
Training script for the dual-branch deepfake detection module.

Architecture: EfficientNet-B4 (spatial) + FFT frequency branch + cross-attention fusion.
Trained on FaceForensics++ (c23 compression), CelebDF-v2, and DFDC.
Uses focal loss to handle the strong class imbalance in DFDC.

Key tricks:
    - Face alignment before crop using dlib/mediapipe
    - Compression-aware augmentation (simulate JPEG re-saves)
    - Frequency-space augmentation (suppress high-freq in real samples)
    - Gradient checkpointing to fit EfficientNet-B4 at batch size 64 on A100

Run:
    python training/train_deepfake.py --config configs/deepfake_config.yaml
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from evaluation.metrics import compute_verification_metrics
from models.deepfake.cnn_transformer import DeepfakeDetector, DeepfakeLoss

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Compression-aware augmentation
# ──────────────────────────────────────────────────────────────────────────────

def random_jpeg_compress(img: np.ndarray, quality_range=(40, 95)) -> np.ndarray:
    """Simulate JPEG re-encoding at random quality."""
    q = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, buf = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def frequency_suppress(img: np.ndarray, keep_fraction: float = 0.8) -> np.ndarray:
    """Zero out high-frequency FFT components in a random channel."""
    ch = random.randint(0, 2)
    plane = img[:, :, ch].astype(np.float32)
    f = np.fft.fft2(plane)
    fshift = np.fft.fftshift(f)
    H, W = plane.shape
    mask = np.zeros_like(fshift, dtype=bool)
    r = int(min(H, W) * keep_fraction / 2)
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    mask[(Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2] = True
    fshift[~mask] = 0
    suppressed = np.fft.ifft2(np.fft.ifftshift(fshift)).real
    img[:, :, ch] = np.clip(suppressed, 0, 255).astype(np.uint8)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Expects structure:
        root/
            real/   <- real face crops (jpg/png)
            fake/   <- deepfake face crops

    Applies compression augmentation and frequency augmentation at training time.
    """

    def __init__(self, root: str, image_size: int = 224, augment: bool = True):
        import torchvision.transforms as T

        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        for label, split in [(0, "real"), (1, "fake")]:
            split_dir = self.root / split
            if split_dir.exists():
                for ext in ("*.jpg", "*.png", "*.jpeg"):
                    for p in split_dir.rglob(ext):
                        self.samples.append((p, label))

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.augment:
            # Compression augmentation (more aggressive for fake samples)
            if random.random() < 0.7:
                quality_range = (30, 80) if label == 1 else (60, 95)
                img = random_jpeg_compress(img, quality_range)

            # Frequency suppression (real images only, to prevent freq shortcuts)
            if label == 0 and random.random() < 0.3:
                img = frequency_suppress(img)

            # Color aug
            if random.random() < 0.5:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img[:, :, 1] = np.clip(img[:, :, 1] * (0.7 + random.random() * 0.6), 0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            # Horizontal flip
            if random.random() < 0.5:
                img = cv2.flip(img, 1)

        # BGR -> RGB -> tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        tensor = self.normalize(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)

    def class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        n_real = max(1, len(labels) - sum(labels))
        n_fake = max(1, sum(labels))
        weights = [1.0 / n_real if lbl == 0 else 1.0 / n_fake for lbl in labels]
        return torch.tensor(weights)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model: DeepfakeDetector, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_scores, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    preds = (scores > 0.5).astype(int)
    acc = float((preds == labels).mean())

    # For deepfake detection, "genuine" = real (0) and "impostor" = fake (1)
    try:
        metrics = compute_verification_metrics(scores, labels)
        auc = metrics.auc
        eer = metrics.eer
    except Exception:
        auc, eer = 0.0, 0.5

    model.train()
    return {"acc": acc, "auc": auc, "eer": eer}


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    console.print(f"Training deepfake detector on [bold]{device}[/bold]")

    # Data
    train_datasets, val_dataset = [], None
    for ds_cfg in cfg.data.get("datasets", []):
        train_root = Path(ds_cfg.root) / "train"
        val_root = Path(ds_cfg.root) / "val"
        if train_root.exists():
            ds = DeepfakeDataset(str(train_root), cfg.data.image_size, augment=True)
            train_datasets.append(ds)
            console.print(f"  {ds_cfg.name}: {len(ds):,} samples")
        else:
            console.print(f"  [dim]{ds_cfg.name}: not found, skipping[/dim]")
        if val_root.exists() and val_dataset is None:
            val_dataset = DeepfakeDataset(str(val_root), cfg.data.image_size, augment=False)

    if not train_datasets:
        console.print("[red]No deepfake datasets found.[/red]")
        return

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(train_datasets)
    all_weights = torch.cat([ds.class_weights() for ds in train_datasets])
    sampler = WeightedRandomSampler(all_weights, len(combined), replacement=True)

    loader = DataLoader(
        combined,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.compute.num_workers,
        pin_memory=cfg.compute.pin_memory,
        persistent_workers=cfg.compute.persistent_workers,
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model
    model = DeepfakeDetector(pretrained=True).to(device)
    if cfg.project.get("gradient_checkpointing", False):
        # EfficientNet-B4 blocks support gradient checkpointing via timm
        try:
            model.spatial_branch.set_grad_checkpointing(True)
            console.print("Gradient checkpointing enabled")
        except AttributeError:
            pass

    if cfg.project.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        console.print("torch.compile enabled")

    criterion = DeepfakeLoss(gamma=cfg.training.focal_gamma, alpha=cfg.training.focal_alpha)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs * len(loader)
    )
    scaler = GradScaler(enabled=cfg.project.mixed_precision)

    wandb.init(project="sentinelid", name="deepfake-detector", config=dict(cfg))
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as prog:
            task = prog.add_task(f"Epoch {epoch+1}/{cfg.training.epochs}", total=len(loader))

            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(enabled=cfg.project.mixed_precision):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1
                prog.advance(task)

                if global_step % 500 == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "step": global_step,
                    })

        avg_loss = epoch_loss / len(loader)
        console.print(
            f"Epoch {epoch+1:3d} | loss: {avg_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s"
        )

        if val_loader and (epoch + 1) % cfg.training.get("eval_every_n_epochs", 3) == 0:
            metrics = evaluate(model, val_loader, device)
            console.print(
                f"  Val -> Acc: {metrics['acc']:.4f} | "
                f"AUC: {metrics['auc']:.4f} | "
                f"EER: {metrics['eer']:.4f}"
            )
            wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"epoch": epoch + 1})
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save(model.state_dict(), ckpt_dir / "deepfake_best.pt")
                console.print(f"  [green]New best AUC: {best_auc:.4f}[/green]")

        torch.save(model.state_dict(), ckpt_dir / "deepfake_latest.pt")

    wandb.finish()
    console.print("[green]Deepfake detection training complete.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deepfake_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
