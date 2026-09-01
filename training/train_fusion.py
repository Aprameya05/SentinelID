"""
Training script for the score fusion module.

Trains ScoreFusionModel on calibrated module outputs.
Requires a held-out calibration set where all 5 module scores are available.

Pipeline:
    1. Run all upstream models in inference mode to generate score vectors
    2. Platt-calibrate each module independently
    3. Train the fusion MLP with BCE + confidence penalty loss

This script handles both steps 1 and 3 (calibration is done inline).

Run:
    python training/train_fusion.py --config configs/fusion_config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.fusion.score_fusion import FusionLoss, PlattCalibrator, ScoreFusionModel

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Score vector dataset
# ──────────────────────────────────────────────────────────────────────────────

class ScoreVectorDataset(Dataset):
    """
    Loads pre-computed score vectors from .npz files.
    Each file has:
        scores: (5,) float32 [liveness, deepfake_real_prob, face_match, au_signal, doc_score]
        label: int (1=genuine, 0=impostor)
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.files = sorted(self.root.rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        scores = torch.from_numpy(data["scores"]).float()
        label = torch.tensor(float(data["label"]), dtype=torch.float32)
        return scores, label


def generate_score_vectors(cfg, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load all upstream models and run inference over a labeled dataset to
    generate score vectors. Returns (score_matrix, labels).

    This is a stub -- in practice you point this at each test split and
    collect outputs. Replace with actual inference calls when models are trained.
    """
    console.print("[yellow]Generating score vectors from upstream models...[/yellow]")
    score_dir = Path(cfg.paths.score_cache_dir)

    if score_dir.exists():
        files = sorted(score_dir.rglob("*.npz"))
        if files:
            scores_list, labels_list = [], []
            for f in files:
                data = np.load(f)
                scores_list.append(data["scores"])
                labels_list.append(float(data["label"]))
            scores = torch.from_numpy(np.stack(scores_list)).float()
            labels = torch.tensor(labels_list).float()
            console.print(f"  Loaded {len(files):,} cached score vectors")
            return scores, labels

    console.print("[yellow]Generating synthetic score vectors...[/yellow]")
    _n = 3000
    _np = __import__("numpy")
    _labels = _np.random.randint(0, 2, _n).astype(_np.float32)
    _scores = _np.where(
        _labels.reshape(-1,1) == 1,
        _np.clip(_np.random.normal(0.8, 0.12, (_n,5)), 0, 1),
        _np.clip(_np.random.normal(0.2, 0.12, (_n,5)), 0, 1),
    ).astype(_np.float32)
    return torch.from_numpy(_scores), torch.from_numpy(_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_scores(
    scores: torch.Tensor,   # (N, 5)
    labels: torch.Tensor,   # (N,)
    module_names: list[str],
) -> tuple[torch.Tensor, list[PlattCalibrator]]:
    """
    Fit one Platt calibrator per module score dimension.
    Returns calibrated scores and list of calibrators.
    """
    calibrators = []
    calibrated_cols = []

    for i, name in enumerate(module_names):
        cal = PlattCalibrator()
        col = scores[:, i].numpy()
        lbl = labels.numpy()
        cal.fit(col, lbl)
        calibrated = cal.transform(col)
        calibrated_cols.append(torch.from_numpy(calibrated).float())
        console.print(f"  Calibrated {name}: mean={calibrated.mean():.3f}")
        calibrators.append(cal)

    return torch.stack(calibrated_cols, dim=1), calibrators


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    console.print(f"Training fusion model on [bold]{device}[/bold]")

    # Load or generate scores
    scores, labels = generate_score_vectors(cfg, device)
    if len(scores) == 0:
        return

    module_names = ["liveness", "deepfake_real", "face_match", "au_signal", "document"]

    # Train/val split
    n_total = len(scores)
    n_val = max(1, int(n_total * 0.15))
    idx = torch.randperm(n_total)
    train_scores = scores[idx[n_val:]]
    train_labels = labels[idx[n_val:]]
    val_scores = scores[idx[:n_val]]
    val_labels = labels[idx[:n_val]]

    # Platt calibration on training set
    cal_train, calibrators = calibrate_scores(train_scores, train_labels, module_names)
    # Apply calibrators to val
    cal_val_cols = []
    for i, cal in enumerate(calibrators):
        cal_val_cols.append(torch.from_numpy(cal.transform(val_scores[:, i].numpy())).float())
    cal_val = torch.stack(cal_val_cols, dim=1)

    train_loader = DataLoader(
        TensorDataset(cal_train, train_labels),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(cal_val, val_labels),
        batch_size=256,
        shuffle=False,
    )

    # Model
    model = ScoreFusionModel(
        n_modules=len(module_names),
        hidden_dim=getattr(cfg.model, "hidden_dim", 64),
    ).to(device)

    criterion = FusionLoss(
        confidence_penalty=cfg.training.get("confidence_penalty", 0.1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    wandb.init(project="sentinelid", name="score-fusion", config=dict(cfg), settings=wandb.Settings(init_timeout=180))
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0

        for score_batch, label_batch in train_loader:
            score_batch = score_batch.to(device)
            label_batch = label_batch.to(device)

            trust_score, logit = model(score_batch)
            loss = criterion(logit, label_batch, trust_score)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels_val = [], []
        with torch.no_grad():
            for score_batch, label_batch in val_loader:
                score_batch = score_batch.to(device)
                label_batch = label_batch.to(device)
                trust_score, logit = model(score_batch)
                val_loss += criterion(logit, label_batch, trust_score).item()
                all_preds.extend(trust_score.cpu().numpy().tolist())
                all_labels_val.extend(label_batch.cpu().numpy().tolist())

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        preds = np.array(all_preds)
        lbl_val = np.array(all_labels_val)
        acc = float(((preds > 0.5).astype(int) == lbl_val.astype(int)).mean())

        console.print(
            f"Epoch {epoch+1:4d} | train: {avg_loss:.4f} | val: {avg_val:.4f} | acc: {acc:.4f}"
        )
        wandb.log({
            "train/loss": avg_loss,
            "val/loss": avg_val,
            "val/acc": acc,
            "epoch": epoch + 1,
        })

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model": model.state_dict(),
                "calibrators": calibrators,
                "module_names": module_names,
            }, ckpt_dir / "fusion_best.pt")
            console.print(f"  [green]New best val loss: {best_val_loss:.4f}[/green]")

    wandb.finish()
    console.print("[green]Fusion training complete.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
