"""
Knowledge distillation script: teacher ensemble -> SentinelEdgeModel (MobileNetV3-Large).

The student learns from soft targets (T=4 KL divergence) from each teacher head,
plus hard label BCE, plus embedding L2 matching for the face recognition head.

Temperature is annealed: T=4 for first third, T=2 for middle, T=1 for final third.

After training, exports to:
    checkpoints/edge_model.onnx    (fp32, legacy exporter)
    checkpoints/edge_model.tflite  (int8, requires onnx2tf)

And benchmarks latency on CPU.

Run:
    python training/distill_edge.py --config configs/distillation_config.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader

from models.behavioral.au_gnn import ActionUnitGNN
from models.edge.distillation import DistillationLoss, EdgeDistiller, SentinelEdgeModel
from models.face_recognition.arcface import ArcFaceModel
from models.liveness.depth_liveness import DepthLivenessModel

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Teacher wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TeacherEnsemble(nn.Module):
    """
    Wraps the three teacher models and returns soft targets for each head.
    All teachers run in inference mode (no grad).
    """

    def __init__(
        self,
        liveness_model: DepthLivenessModel,
        face_model: ArcFaceModel,
        au_model: ActionUnitGNN,
    ):
        super().__init__()
        self.liveness = liveness_model
        self.face = face_model
        self.au = au_model

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, landmarks: torch.Tensor | None = None):
        """
        images: (B, 3, H, W)
        landmarks: (B, 68, 2) optional
        Returns soft targets dict.
        """
        liveness_out = self.liveness(images)
        liveness_logit = liveness_out["liveness_logit"]

        face_embed = self.face.embed(images)   # (B, 512)

        if landmarks is not None:
            au_pred, _ = self.au(landmarks)
            au_signal = au_pred.mean(dim=1)
        else:
            au_signal = torch.zeros(images.shape[0], device=images.device)

        return {
            "liveness_logit": liveness_logit,
            "face_embed": face_embed,
            "au_signal": au_signal,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DistillDataset(torch.utils.data.Dataset):
    """
    Loads image + liveness label.
    Expects root/<live|spoof>/**/*.{jpg,png}
    """

    def __init__(self, root: str, image_size: int = 224, augment: bool = True):
        import torchvision.transforms as T

        self.root = Path(root)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []

        for label, split in [(1, "live"), (0, "spoof")]:
            split_dir = self.root / split
            if split_dir.exists():
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for p in split_dir.rglob(ext):
                        self.samples.append((p, label))

        aug_list = []
        if augment:
            aug_list = [
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                T.RandomGrayscale(p=0.05),
                T.RandomErasing(p=0.1),
            ]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            *aug_list,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(float(label))


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    console.print(f"Knowledge distillation on [bold]{device}[/bold]")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load teachers from checkpoints ──────────────────────────────────────
    def load_or_skip(model_cls, ckpt_name, *args, **kwargs):
        path = ckpt_dir / ckpt_name
        m = model_cls(*args, **kwargs)
        if path.exists():
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            console.print(f"  Loaded teacher: {ckpt_name}")
        else:
            console.print(f"  [dim]Teacher checkpoint not found: {ckpt_name} (random init)[/dim]")
        return m.to(device).eval()

    liveness_teacher = load_or_skip(DepthLivenessModel, "liveness_best.pt")
    face_teacher = load_or_skip(
        ArcFaceModel, "face_model.pt",
        num_classes=cfg.face.get("num_classes", 85742),
        embedding_dim=512,
        backbone="iresnet100",
    )
    au_teacher = load_or_skip(ActionUnitGNN, "behavioral_best.pt")

    teacher = TeacherEnsemble(liveness_teacher, face_teacher, au_teacher)

    # ── Student ─────────────────────────────────────────────────────────────
    student = SentinelEdgeModel(
        liveness_out=1,
        face_embed_dim=cfg.student.get("face_embed_dim", 256),
        au_out=cfg.student.get("au_out", 1),
    ).to(device)

    if cfg.project.get("compile_model", False) and hasattr(torch, "compile"):
        student = torch.compile(student)
        console.print("torch.compile enabled for student")

    # ── Data ────────────────────────────────────────────────────────────────
    train_datasets = []
    for ds_cfg in cfg.data.get("datasets", []):
        root = Path(ds_cfg.root) / "train"
        if root.exists():
            ds = DistillDataset(str(root), cfg.data.image_size, augment=True)
            train_datasets.append(ds)
            console.print(f"  {ds_cfg.name}: {len(ds):,} samples")
        else:
            console.print(f"  [dim]{ds_cfg.name}: {root} not found, skipping[/dim]")

    if not train_datasets:
        console.print("[red]No datasets found for distillation.[/red]")
        return

    combined = ConcatDataset(train_datasets)
    loader = DataLoader(
        combined,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
        pin_memory=cfg.compute.pin_memory,
        drop_last=True,
    )

    # ── Loss / Optim ─────────────────────────────────────────────────────────
    criterion = DistillationLoss(
        temperature=cfg.training.get("temperature", 4.0),
        alpha=cfg.training.get("alpha", 0.7),
        embedding_weight=cfg.training.get("embedding_weight", 0.3),
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = cfg.training.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler("cuda", enabled=cfg.project.get("mixed_precision", True))

    wandb.init(project="sentinelid", name="edge-distillation", config=dict(cfg))
    global_step = 0
    epochs = cfg.training.epochs

    # Temperature schedule: T=4 → T=2 → T=1
    def get_temperature(epoch: int) -> float:
        if epoch < epochs // 3:
            return 4.0
        elif epoch < 2 * epochs // 3:
            return 2.0
        return 1.0

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Anneal distillation temperature
        T = get_temperature(epoch)
        criterion.set_temperature(T)

        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as prog:
            task = prog.add_task(f"Epoch {epoch+1}/{epochs} (T={T})", total=len(loader))

            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                teacher_out = teacher(images)

                with autocast("cuda", enabled=cfg.project.get("mixed_precision", True)):
                    student_out = student(images)
                    loss = criterion(
                        student_liveness=student_out["logits_liveness"],
                        student_embedding=student_out["face_embed"],
                        teacher_liveness_logit=teacher_out["liveness_logit"],
                        teacher_embedding=teacher_out["face_embed"],
                        hard_labels=labels,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1
                prog.advance(task)

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        console.print(
            f"Epoch {epoch+1:3d}/{epochs} | loss: {avg_loss:.4f} | T={T} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
        )
        wandb.log({"train/loss": avg_loss, "temperature": T, "epoch": epoch + 1})

        torch.save(student.state_dict(), ckpt_dir / "edge_model.pt")

    # ── Export ───────────────────────────────────────────────────────────────
    console.print("\n[bold]Exporting edge model...[/bold]")

    # unwrap from torch.compile if needed
    raw_student = student._orig_mod if hasattr(student, "_orig_mod") else student
    distiller = EdgeDistiller(raw_student)

    img_size = cfg.data.image_size
    onnx_path = str(ckpt_dir / "edge_model.onnx")
    distiller.export_onnx(onnx_path, input_size=(1, 3, img_size, img_size))

    tflite_path = str(ckpt_dir / "edge_model.tflite")
    distiller.export_tflite(onnx_path, tflite_path)

    latency = distiller.benchmark(num_runs=200, input_size=(1, 3, img_size, img_size))
    console.print(
        f"\n[bold green]Edge model latency (CPU):[/bold green] "
        f"mean={latency['mean_ms']:.1f}ms  "
        f"p95={latency['p95_ms']:.1f}ms  "
        f"p99={latency['p99_ms']:.1f}ms"
    )
    wandb.log({"latency/mean_ms": latency["mean_ms"], "latency/p95_ms": latency["p95_ms"]})

    wandb.finish()
    console.print("[green]Distillation complete.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distillation_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
