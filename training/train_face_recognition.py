"""
Training script for the ArcFace face recognition backbone.

Phase 1 of the SentinelID compute plan — this is the longest single job
(~38h on a single A100). Uses torch.compile, mixed precision, and gradient
checkpointing to maximise throughput.

Run:
    python training/train_face_recognition.py --config configs/arcface_config.yaml
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import wandb

from models.face_recognition.arcface import ArcFaceModel

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset stub (replace with actual loaders in data/datasets/)
# ──────────────────────────────────────────────────────────────────────────────

class FaceDataset(torch.utils.data.Dataset):
    """
    Wrapper around an image folder dataset for face recognition.
    Each subfolder = one identity.
    """

    def __init__(self, root: str, transform=None):
        from torchvision.datasets import ImageFolder
        self.dataset = ImageFolder(root=root, transform=transform)
        self.num_classes = len(self.dataset.classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def build_transform(image_size: int = 112):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.RandomErasing(p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# LFW evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def eval_lfw(model: ArcFaceModel, lfw_root: str, device: torch.device) -> dict[str, float]:
    """Evaluate on LFW pairs. Returns TAR@FAR=1e-3 and verification AUC."""
    from sklearn.metrics import roc_auc_score, roc_curve
    import numpy as np

    lfw_path = Path(lfw_root)
    pairs_file = lfw_path / "pairs.txt"
    if not pairs_file.exists():
        return {"lfw_auc": 0.0, "lfw_tar_far1e3": 0.0}

    transform = build_transform(112)
    model.eval()
    sims, labels = [], []

    with open(pairs_file) as f:
        lines = f.readlines()[1:]  # skip header

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:  # positive pair: name n1 n2
            name, n1, n2 = parts[0], int(parts[1]), int(parts[2])
            p1 = lfw_path / name / f"{name}_{n1:04d}.jpg"
            p2 = lfw_path / name / f"{name}_{n2:04d}.jpg"
            label = 1
        elif len(parts) == 4:  # negative pair: name1 n1 name2 n2
            n1, n2 = parts[0], parts[2]
            p1 = lfw_path / n1 / f"{n1}_{int(parts[1]):04d}.jpg"
            p2 = lfw_path / n2 / f"{n2}_{int(parts[3]):04d}.jpg"
            label = 0
        else:
            continue

        try:
            from PIL import Image
            i1 = transform(Image.open(p1).convert("RGB")).unsqueeze(0).to(device)
            i2 = transform(Image.open(p2).convert("RGB")).unsqueeze(0).to(device)
            e1 = model.embed(i1).cpu().numpy()[0]
            e2 = model.embed(i2).cpu().numpy()[0]
            sim = float(np.dot(e1, e2))
            sims.append(sim)
            labels.append(label)
        except Exception:
            continue

    if not sims:
        return {"lfw_auc": 0.0, "lfw_tar_far1e3": 0.0}

    sims = np.array(sims)
    labels = np.array(labels)
    auc = roc_auc_score(labels, sims)
    fpr, tpr, _ = roc_curve(labels, sims)
    # TAR at FAR = 1e-3
    tar_idx = np.searchsorted(fpr, 1e-3)
    tar = tpr[min(tar_idx, len(tpr) - 1)]

    return {"lfw_auc": float(auc), "lfw_tar_far1e3": float(tar)}


# ──────────────────────────────────────────────────────────────────────────────
# Per-demographic bias audit
# ──────────────────────────────────────────────────────────────────────────────

def audit_demographic_fairness(model, audit_dataset, device) -> dict[str, dict]:
    """
    Compute FMR and FNMR per demographic group.
    Requires audit_dataset to return (image, label, demographic_group).
    """
    console.print("[yellow]Running demographic fairness audit...[/yellow]")
    results = {}
    # Stub: real implementation requires labelled demographic test set
    # (e.g. Diversity in Faces dataset, BFW dataset)
    console.print("[dim]Fairness audit requires labelled demographic dataset.[/dim]")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    console.print(f"Training on [bold]{device}[/bold]")

    # Data
    transform = build_transform(cfg.data.image_size)
    datasets = []
    total_classes = 0

    for ds_cfg in cfg.data.datasets:
        if Path(ds_cfg.root).exists():
            ds = FaceDataset(ds_cfg.root, transform=transform)
            datasets.append(ds)
            total_classes = max(total_classes, ds.num_classes)
            console.print(f"  {ds_cfg.name}: {len(ds):,} samples, {ds.num_classes:,} classes")
        else:
            console.print(f"  [dim]{ds_cfg.name}: path not found, skipping[/dim]")

    if not datasets:
        console.print("[red]No datasets found. Check data paths in config.[/red]")
        return

    combined = ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
        pin_memory=cfg.compute.pin_memory,
        persistent_workers=cfg.compute.persistent_workers,
        prefetch_factor=cfg.compute.prefetch_factor,
        drop_last=True,
    )

    # Model
    model = ArcFaceModel(
        num_classes=total_classes,
        embedding_dim=cfg.model.embedding_dim,
        backbone=cfg.model.backbone,
    ).to(device)

    if cfg.project.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        console.print("torch.compile enabled")

    # Optimiser
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs * len(loader)
    )
    scaler = GradScaler(enabled=cfg.project.mixed_precision)

    # W&B
    wandb.init(project="sentinelid", name="arcface-face-recognition", config=dict(cfg))

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
                    loss = model(images, labels)

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
        elapsed = time.time() - t0
        console.print(
            f"Epoch {epoch+1:3d} | loss: {avg_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
        )

        # Save checkpoint
        torch.save(model.state_dict(), ckpt_dir / "face_model.pt")

        # Periodic LFW evaluation
        if (epoch + 1) % cfg.training.get("eval_every_n_epochs", 5) == 0:
            lfw_root = Path(cfg.paths.data_root) / "lfw"
            if lfw_root.exists():
                metrics = eval_lfw(model, str(lfw_root), device)
                console.print(f"  LFW AUC: {metrics['lfw_auc']:.4f} | TAR@FAR1e-3: {metrics['lfw_tar_far1e3']:.4f}")
                wandb.log({**metrics, "epoch": epoch + 1})

    wandb.finish()
    console.print("[green]Training complete.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/arcface_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
