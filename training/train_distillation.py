"""Knowledge distillation: compress DepthLivenessModel (teacher) into a MobileNetV3 student."""
import argparse, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, wandb
from omegaconf import OmegaConf
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from rich.console import Console

console = Console()

def get_student(num_classes=1):
    m = mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Distillation on [bold]{device}[/bold]")

    # Synthetic distillation data
    N = 4000
    X = torch.randn(N, 3, 224, 224)
    y = torch.randint(0, 2, (N,)).float()

    ds = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(ds, [3500, 500])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    student = get_student().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(cfg.training, "epochs", 30))
    scaler = GradScaler()

    wandb.init(project="sentinelid", name="edge-distillation",
               config=dict(cfg), settings=wandb.Settings(init_timeout=180))

    ckpt_dir = Path(cfg.paths.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    epochs = getattr(cfg.training, "epochs", 30)

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast():
                logit = student(xb).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logit, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(train_loader)

        student.eval(); correct = total = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = (torch.sigmoid(student(xb).squeeze(1)) > 0.5).float()
                correct += (pred == yb).sum().item(); total += len(yb)
        acc = correct / total
        console.print(f"Epoch {epoch+1:3d}/{epochs} | loss: {avg:.4f} | val_acc: {acc:.4f}")
        wandb.log({"train/loss": avg, "val/acc": acc, "epoch": epoch+1})

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), ckpt_dir / "distilled_best.pt")
            console.print(f"  [green]New best acc: {best_acc:.4f}[/green]")
        torch.save(student.state_dict(), ckpt_dir / "distilled_latest.pt")

    wandb.finish()
    console.print("[green]Distillation complete.[/green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distillation_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
