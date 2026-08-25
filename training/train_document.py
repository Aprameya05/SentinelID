"""
Training script for the document intelligence module.

Trains DocumentIntelligenceModel on MIDV-500 + MIDV-2020.
Uses a BERT tokenizer for text token embedding.

Pipeline:
    Image patches (ViT) + WordPiece tokens + 2D spatial bbox embeddings
    -> 12-layer transformer encoder
    -> Document type classifier + BIO field extraction + Forgery detection

Loss: DocumentLoss = 0.5 * CE(doc_type) + 1.0 * CE(field_BIO) + 1.5 * focal(forgery)

Run:
    python training/train_document.py --config configs/document_config.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader

from data.datasets.document import MIDV500Dataset, MIDV2020Dataset
from models.document.layout_intelligence import DocumentIntelligenceModel, DocumentLoss

console = Console()


def get_tokenizer(cfg):
    try:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        return tokenizer
    except Exception as e:
        console.print(f"[yellow]Tokenizer load failed ({e}); using token_ids=zeros fallback.[/yellow]")
        return None


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    """Collate variable-length token sequences by stacking fixed-length tensors."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "token_ids": torch.stack([b["token_ids"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "doc_type_label": torch.stack([b["doc_type_label"] for b in batch]),
        "forgery_label": torch.stack([b["forgery_label"] for b in batch]),
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    n_forgery_correct = 0

    for batch in loader:
        image = batch["image"].to(device)
        token_ids = batch["token_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        doc_type_labels = batch["doc_type_label"].to(device)
        forgery_labels = batch["forgery_label"].to(device)

        outputs = model(image, token_ids, bbox, attention_mask)
        loss = criterion(
            outputs,
            doc_type_labels=doc_type_labels,
            field_labels={},
            forgery_labels=forgery_labels,
        )
        total_loss += loss.item()

        pred_type = outputs["doc_type_logits"].argmax(dim=1)
        n_correct += (pred_type == doc_type_labels).sum().item()
        n_total += len(doc_type_labels)

        forgery_pred = (outputs["forgery_prob"] > 0.5).float()
        n_forgery_correct += (forgery_pred == forgery_labels).sum().item()

    model.train()
    return {
        "loss": total_loss / max(1, len(loader)),
        "doc_type_acc": n_correct / max(1, n_total),
        "forgery_acc": n_forgery_correct / max(1, n_total),
    }


def train(cfg):
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    console.print(f"Training document intelligence on [bold]{device}[/bold]")

    tokenizer = get_tokenizer(cfg)

    # Data
    train_datasets, val_dataset = [], None
    dataset_classes = {"midv500": MIDV500Dataset, "midv2020": MIDV2020Dataset}

    for ds_cfg in cfg.data.get("datasets", []):
        cls_name = ds_cfg.name.replace("-", "").lower()
        DatasetCls = dataset_classes.get(cls_name, MIDV500Dataset)
        train_root = Path(ds_cfg.root) / "train"
        val_root = Path(ds_cfg.root) / "val"

        if train_root.exists():
            ds = DatasetCls(str(train_root), augment=True, tokenizer=tokenizer)
            train_datasets.append(ds)
            console.print(f"  {ds_cfg.name}: {len(ds):,} samples")
        elif Path(ds_cfg.root).exists():
            # No train/val split; use whole dataset
            ds = DatasetCls(str(ds_cfg.root), augment=True, tokenizer=tokenizer)
            train_datasets.append(ds)
            console.print(f"  {ds_cfg.name}: {len(ds):,} samples (no split)")
        else:
            console.print(f"  [dim]{ds_cfg.name}: not found, skipping[/dim]")
            continue

        if val_root.exists() and val_dataset is None:
            val_dataset = DatasetCls(str(val_root), augment=False, tokenizer=tokenizer)

    if not train_datasets:
        console.print("[red]No document datasets found.[/red]")
        return

    combined = ConcatDataset(train_datasets)
    loader = DataLoader(
        combined,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
        pin_memory=cfg.compute.pin_memory,
        persistent_workers=cfg.compute.persistent_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    # Model
    model = DocumentIntelligenceModel(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        image_size=cfg.data.image_size,
        patch_size=cfg.model.patch_size,
        max_seq_len=cfg.data.max_seq_len,
    ).to(device)

    criterion = DocumentLoss(
        field_weight=cfg.training.field_weight,
        doc_type_weight=cfg.training.doc_type_weight,
        forgery_weight=cfg.training.forgery_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = cfg.training.epochs * len(loader)
    warmup_steps = cfg.training.get("warmup_steps", 1000)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.project.mixed_precision)

    wandb.init(project="sentinelid", name="document-intelligence", config=dict(cfg))
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as prog:
            task = prog.add_task(f"Epoch {epoch+1}/{cfg.training.epochs}", total=len(loader))

            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                token_ids = batch["token_ids"].to(device, non_blocking=True)
                bbox = batch["bbox"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                doc_type_labels = batch["doc_type_label"].to(device, non_blocking=True)
                forgery_labels = batch["forgery_label"].to(device, non_blocking=True)

                with autocast(enabled=cfg.project.mixed_precision):
                    outputs = model(image, token_ids, bbox, attention_mask)
                    loss = criterion(
                        outputs,
                        doc_type_labels=doc_type_labels,
                        field_labels={},
                        forgery_labels=forgery_labels,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1
                prog.advance(task)

                if global_step % 200 == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "step": global_step,
                    })

        avg_loss = epoch_loss / len(loader)
        console.print(
            f"Epoch {epoch+1:3d} | loss: {avg_loss:.4f} | {time.time()-t0:.0f}s"
        )
        wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch + 1})

        if val_loader and (epoch + 1) % cfg.training.get("eval_every_n_epochs", 3) == 0:
            metrics = evaluate(model, val_loader, criterion, device)
            console.print(
                f"  Val -> loss: {metrics['loss']:.4f} | "
                f"doc_type_acc: {metrics['doc_type_acc']:.3f} | "
                f"forgery_acc: {metrics['forgery_acc']:.3f}"
            )
            wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"epoch": epoch + 1})

            if metrics["doc_type_acc"] > best_acc:
                best_acc = metrics["doc_type_acc"]
                torch.save(model.state_dict(), ckpt_dir / "document_best.pt")
                console.print(f"  [green]New best doc type acc: {best_acc:.3f}[/green]")

        torch.save(model.state_dict(), ckpt_dir / "document_latest.pt")

    wandb.finish()
    console.print("[green]Document intelligence training complete.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/document_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
