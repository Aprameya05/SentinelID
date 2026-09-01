"""
Training script for the depth + liveness detection module.

Trains DepthLivenessModel on a mix of spoof datasets:
    - NUAA Imposter (print attacks)
    - 3DMADBv2 (3D mask attacks)
    - CASIA-SURF (depth + IR + RGB multimodal)
    - SiW (diverse spoof types)

Optimizes LivenessLoss = BCE(liveness) + BerHu(depth) + contrastive margin.
Reports ISO 30107-3 metrics: ACER, APCER, BPCER at EER threshold.

Run:
    python training/train_liveness.py --config configs/liveness_config.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from evaluation.metrics import compute_liveness_metrics
from models.liveness.depth_liveness import DepthLivenessModel, LivenessLoss

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class LivenessDataset(Dataset):
    """
    Expects a root directory with structure:
        root/
            live/    <- genuine face images
            spoof/   <- attack presentation images

    Returns (image_tensor, depth_tensor, liveness_label).
    depth_tensor is all-zeros when ground-truth depth is unavailable
    (the BerHu loss is masked out for zero-depth samples).
    """

    def __init__(self, root: str, image_size: int = 256, augment: bool = True):
        import torchvision.transforms as T

        self.root = Path(root)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []

        for label, split in [(1, "live"), (0, "spoof")]:
            split_dir = self.root / split
            if split_dir.exists():
                for p in split_dir.rglob("*.jpg"):
                    self.samples.append((p, label))
                for p in split_dir.rglob("*.png"):
                    self.samples.append((p, label))

        if augment:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                T.RandomErasing(p=0.2),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        # Depth ground truth: load adjacent .npy if exists, else zeros
        depth_path = path.with_suffix(".npy")
        if depth_path.exists():
            depth = torch.from_numpy(np.load(depth_path)).float().unsqueeze(0)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(self.image_size // 4, self.image_size // 4)
            ).squeeze(0)
            has_depth = torch.ones(1)
        else:
            depth = torch.zeros(1, self.image_size // 4, self.image_size // 4)
            has_depth = torch.zeros(1)

        return img_t, depth, torch.tensor(label, dtype=torch.float32), has_depth

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for balanced sampling."""
        labels = [s[1] for s in self.samples]
        n_live = sum(labels)
        n_spoof = len(labels) - n_live
        weights = [1.0 / n_live if lbl == 1 else 1.0 / n_spoof for lbl in labels]
        return torch.tensor(weights)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model: DepthLivenessModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_scores, all_labels = [], []

    for images, depth_gt, labels, has_depth in loader:
        images = images.to(device)
        out = model(images)
        scores = torch.sigmoid(out["liveness_logit"]).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.numpy().tolist())

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    metrics = compute_liveness_metrics(scores_arr, labels_arr)
    model.train()
    return {
        "acer": metrics.acer,
        "apcer": metrics.apcer,
        "bpcer": metrics.bpcer,
        "auc": metrics.auc,
        "eer": metrics.eer,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    console.print(f"Training liveness model on [bold]{device}[/bold]")

    # Data
    train_datasets, val_dataset = [], None
    for ds_cfg in cfg.data.datasets:
        train_root = Path(ds_cfg.root) / "train"
        val_root = Path(ds_cfg.root) / "val"
        if train_root.exists():
            ds = LivenessDataset(str(train_root), cfg.data.image_size, augment=True)
            train_datasets.append(ds)
            console.print(f"  {ds_cfg.name}: {len(ds):,} samples")
        else:
            console.print(f"  [dim]{ds_cfg.name}: not found, skipping[/dim]")
        if val_root.exists() and val_dataset is None:
            val_dataset = LivenessDataset(str(val_root), cfg.data.image_size, augment=False)

    if not train_datasets:
        console.print("[red]No datasets found.[/red]")
        return

    # Concatenate and build weighted sampler for class balance
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(train_datasets)

    # Build combined weights
    all_weights = []
    for ds in train_datasets:
        all_weights.append(ds.class_weights())
    all_weights = torch.cat(all_weights)
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
    model = DepthLivenessModel(backbone=cfg.model.backbone).to(device)
    if cfg.project.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        console.print("torch.compile enabled")

    criterion = LivenessLoss(
        liveness_weight=getattr(cfg.training, 'bce_weight', 1.0),
        depth_weight=getattr(cfg.training, 'depth_weight', 0.5),
        contrastive_weight=getattr(cfg.training, 'contrastive_weight', 0.1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        steps_per_epoch=len(loader),
        epochs=cfg.training.epochs,
        pct_start=0.1,
    )
    scaler = GradScaler(enabled=cfg.project.mixed_precision)

    wandb.init(project="sentinelid", name="depth-liveness", config=dict(cfg))

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acer = float("inf")

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as prog:
            task = prog.add_task(f"Epoch {epoch+1}/{cfg.training.epochs}", total=len(loader))

            for images, depth_gt, liveness_labels, has_depth in loader:
                images = images.to(device, non_blocking=True)
                depth_gt = depth_gt.to(device, non_blocking=True)
                liveness_labels = liveness_labels.to(device, non_blocking=True)

                with autocast(enabled=cfg.project.mixed_precision):
                    outputs = model(images)
                    loss = criterion(
                        outputs["liveness_logit"],
                        outputs["depth_map"],
                        depth_gt,
                        liveness_labels,
                        has_depth.to(device),
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_loss += loss.item()
                prog.advance(task)

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        console.print(
            f"Epoch {epoch+1:3d} | loss: {avg_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
        )
        wandb.log({"train/loss": avg_loss, "epoch": epoch + 1})

        # Validation
        if val_loader and (epoch + 1) % cfg.training.get("eval_every_n_epochs", 5) == 0:
            metrics = evaluate(model, val_loader, device)
            console.print(
                f"  Val -> ACER: {metrics['acer']:.4f} | "
                f"APCER: {metrics['apcer']:.4f} | "
                f"BPCER: {metrics['bpcer']:.4f} | "
                f"AUC: {metrics['auc']:.4f}"
            )
            wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"epoch": epoch + 1})

            if metrics["acer"] < best_acer:
                best_acer = metrics["acer"]
                torch.save(model.state_dict(), ckpt_dir / "liveness_best.pt")
                console.print(f"  [green]New best ACER: {best_acer:.4f}[/green]")

        torch.save(model.state_dict(), ckpt_dir / "liveness_latest.pt")

    wandb.finish()
    console.print("[green]Liveness training complete.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/liveness_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
