"""
Training script for the behavioral biometrics module (AU-GNN + gaze).

Trains ActionUnitGNN on:
    - DISFA (spontaneous AU intensities, 12 AUs, video frames)
    - BP4D (22 AUs, posed + spontaneous)
    - GazeCapture / MPIIGaze for the gaze regression head

The graph topology uses 68 facial landmarks with anatomical adjacency.
Loss: smooth L1 per AU (with per-AU weights from DISFA class frequency)
      + L2 gaze regression.

Run:
    python training/train_behavioral.py --config configs/behavioral_config.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models.behavioral.au_gnn import ActionUnitGNN, AULoss, GazeRegressionHead

console = Console()

# DISFA AU indices and approximate positive-sample frequencies
DISFA_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
DISFA_FREQ = [0.10, 0.09, 0.22, 0.05, 0.18, 0.08, 0.35, 0.07, 0.12, 0.08, 0.45, 0.40]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AUDataset(Dataset):
    """
    Expects preprocessed .npz files with:
        landmarks: (68, 2) float32  -- normalized to [0, 1]
        au_intensities: (n_aus,) float32  -- in [0, 5]
        eye_crop: (3, 64, 64) float32  -- pre-cropped and normalized

    Each .npz = one frame.
    """

    def __init__(self, root: str, n_aus: int = 12):
        self.root = Path(root)
        self.n_aus = n_aus
        self.files = sorted(self.root.rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=False)
        landmarks = torch.from_numpy(data["landmarks"]).float()     # (68, 2)
        au_labels = torch.from_numpy(data["au_intensities"]).float()[:self.n_aus]
        eye_crop = torch.from_numpy(data["eye_crop"]).float()       # (3, 64, 64)
        has_gaze = torch.tensor(1.0 if "gaze_vector" in data else 0.0)
        gaze = torch.from_numpy(data["gaze_vector"]).float() if "gaze_vector" in data else torch.zeros(3)
        return landmarks, au_labels, eye_crop, gaze, has_gaze


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    model: ActionUnitGNN,
    gaze_head: GazeRegressionHead,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    gaze_head.eval()

    au_maes, gaze_errors = [], []

    for landmarks, au_labels, eye_crops, gaze_gt, has_gaze in loader:
        landmarks = landmarks.to(device)
        au_labels = au_labels.to(device)
        eye_crops = eye_crops.to(device)
        gaze_gt = gaze_gt.to(device)
        has_gaze = has_gaze.to(device)

        au_pred, _ = model(landmarks)
        au_mae = (au_pred - au_labels).abs().mean().item()
        au_maes.append(au_mae)

        gaze_mask = has_gaze > 0.5
        if gaze_mask.any():
            gaze_pred = gaze_head(eye_crops[gaze_mask])
            g_err = (gaze_pred - gaze_gt[gaze_mask]).norm(dim=1).mean().item()
            gaze_errors.append(g_err)

    model.train()
    gaze_head.train()
    return {
        "au_mae": float(np.mean(au_maes)),
        "gaze_error": float(np.mean(gaze_errors)) if gaze_errors else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    console.print(f"Training AU-GNN on [bold]{device}[/bold]")

    # Data
    train_datasets, val_dataset = [], None
    for ds_cfg in cfg.data.get("datasets", []):
        train_root = Path(ds_cfg.root) / "train"
        val_root = Path(ds_cfg.root) / "val"
        if train_root.exists():
            ds = AUDataset(str(train_root), n_aus=cfg.model.n_aus)
            train_datasets.append(ds)
            console.print(f"  {ds_cfg.name}: {len(ds):,} frames")
        else:
            console.print(f"  [dim]{ds_cfg.name}: not found, skipping[/dim]")
        if val_root.exists() and val_dataset is None:
            val_dataset = AUDataset(str(val_root), n_aus=cfg.model.n_aus)

    if not train_datasets:
        console.print("[red]No AU datasets found.[/red]")
        return

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(train_datasets)
    loader = DataLoader(
        combined,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
        pin_memory=cfg.compute.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Models
    au_weights = torch.tensor(
        [1.0 / f for f in DISFA_FREQ[:cfg.model.n_aus]], dtype=torch.float32
    ).to(device)
    au_weights = au_weights / au_weights.sum() * cfg.model.n_aus  # normalize

    model = ActionUnitGNN(
        n_landmarks=68,
        node_feat_dim=cfg.model.node_feat_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        n_aus=cfg.model.n_aus,
    ).to(device)

    gaze_head = GazeRegressionHead(in_channels=3, hidden_dim=128).to(device)

    criterion = AULoss(au_weights=au_weights)
    gaze_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(gaze_head.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs * len(loader)
    )
    scaler = GradScaler(enabled=cfg.project.mixed_precision)

    wandb.init(project="sentinelid", name="au-gnn-behavioral", config=dict(cfg))
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    gaze_weight = cfg.training.get("gaze_weight", 0.5)

    for epoch in range(cfg.training.epochs):
        model.train()
        gaze_head.train()
        epoch_loss = 0.0
        t0 = time.time()

        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as prog:
            task = prog.add_task(f"Epoch {epoch+1}/{cfg.training.epochs}", total=len(loader))

            for landmarks, au_labels, eye_crops, gaze_gt, has_gaze in loader:
                landmarks = landmarks.to(device, non_blocking=True)
                au_labels = au_labels.to(device, non_blocking=True)
                eye_crops = eye_crops.to(device, non_blocking=True)
                gaze_gt = gaze_gt.to(device, non_blocking=True)
                has_gaze = has_gaze.to(device, non_blocking=True)

                with autocast(enabled=cfg.project.mixed_precision):
                    au_pred, _ = model(landmarks)
                    au_loss = criterion(au_pred, au_labels)

                    gaze_mask = has_gaze > 0.5
                    if gaze_mask.any():
                        gaze_pred = gaze_head(eye_crops[gaze_mask])
                        gaze_loss = gaze_criterion(gaze_pred, gaze_gt[gaze_mask])
                    else:
                        gaze_loss = torch.tensor(0.0, device=device)

                    loss = au_loss + gaze_weight * gaze_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(gaze_head.parameters()), max_norm=5.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_loss += loss.item()
                prog.advance(task)

        avg_loss = epoch_loss / len(loader)
        console.print(
            f"Epoch {epoch+1:3d} | loss: {avg_loss:.4f} | {time.time()-t0:.0f}s"
        )
        wandb.log({"train/loss": avg_loss, "epoch": epoch + 1})

        if val_loader and (epoch + 1) % cfg.training.get("eval_every_n_epochs", 5) == 0:
            metrics = evaluate(model, gaze_head, val_loader, device)
            console.print(
                f"  Val -> AU MAE: {metrics['au_mae']:.4f} | "
                f"Gaze error: {metrics['gaze_error']:.4f}"
            )
            wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"epoch": epoch + 1})
            if metrics["au_mae"] < best_mae:
                best_mae = metrics["au_mae"]
                torch.save({
                    "au_gnn": model.state_dict(),
                    "gaze_head": gaze_head.state_dict(),
                }, ckpt_dir / "behavioral_best.pt")
                console.print(f"  [green]New best AU MAE: {best_mae:.4f}[/green]")

        torch.save({
            "au_gnn": model.state_dict(),
            "gaze_head": gaze_head.state_dict(),
        }, ckpt_dir / "behavioral_latest.pt")

    wandb.finish()
    console.print("[green]Behavioral biometrics training complete.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavioral_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
