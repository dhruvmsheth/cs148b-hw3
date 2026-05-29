"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def eval_accuracy(model: nn.Module, head: nn.Module, loader, device: torch.device) -> float:
    """Compute classification accuracy on a dataloader."""
    model.eval()
    head.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        feats = model(images)
        logits = head(feats)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    run_name = f"resisc_{args.method}_rank{args.rank}"
    if args.wandb:
        import wandb
        wandb.init(entity="dhruvsheth", project="cs148b-hw3", name=run_name, config={  # noqa
            "method": args.method, "rank": args.rank, "alpha": args.alpha, **cfg
        })

    from vlm.data import build_resisc45_loaders
    from basics.vit import ViT

    ckpt = torch.load(args.pretrained, map_location="cpu")
    vit_config = ckpt["vit_config"]

    train_dl, test_dl = build_resisc45_loaders(
        img_size=vit_config["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    vit = ViT(**vit_config)
    vit.load_state_dict(ckpt["vit_state"])
    num_classes = cfg["num_classes"]
    head = nn.Linear(vit_config["d_model"], num_classes)

    method_lr = cfg["methods"][args.method]["lr"]
    ocfg = cfg["optim"]

    if args.method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad_(False)
        vit.to(device)
        head.to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=method_lr,
            weight_decay=ocfg["weight_decay"], betas=tuple(ocfg["betas"]),
        )

    elif args.method == "lora":
        from basics.lora import apply_lora_to_attention
        vit = apply_lora_to_attention(vit, args.rank, args.alpha)
        vit.to(device)
        head.to(device)
        trainable = [p for p in vit.parameters() if p.requires_grad] + list(head.parameters())
        optimizer = torch.optim.AdamW(
            trainable, lr=method_lr,
            weight_decay=ocfg["weight_decay"], betas=tuple(ocfg["betas"]),
        )

    else:  # full_ft
        for p in vit.parameters():
            p.requires_grad_(True)
        vit.to(device)
        head.to(device)
        optimizer = torch.optim.AdamW(
            list(vit.parameters()) + list(head.parameters()),
            lr=method_lr,
            weight_decay=ocfg["weight_decay"], betas=tuple(ocfg["betas"]),
        )

    num_epochs = cfg["train"]["num_epochs"]
    total_steps = num_epochs * len(train_dl)
    scheduler = get_cosine_schedule_with_warmup(optimizer, ocfg["warmup_steps"], total_steps)

    trainable_params = sum(p.numel() for p in list(vit.parameters()) + list(head.parameters()) if p.requires_grad)
    total_params = sum(p.numel() for p in list(vit.parameters()) + list(head.parameters()))
    print(f"Method={args.method} trainable={trainable_params:,}/{total_params:,}")

    criterion = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_every = cfg["train"]["log_every"]
    global_step = 0
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        vit.train()
        head.train()
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            feats = vit(images)
            logits = head(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in list(vit.parameters()) + list(head.parameters()) if p.requires_grad],
                1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % log_every == 0:
                print(f"epoch={epoch+1} step={global_step} loss={loss.item():.4f}")
                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item()}, step=global_step)

        acc = eval_accuracy(vit, head, test_dl, device)
        print(f"Epoch {epoch+1}/{num_epochs} test_acc={acc:.4f}")
        if args.wandb:
            import wandb
            wandb.log({"test/acc": acc}, step=global_step)
        if acc > best_acc:
            best_acc = acc

    wall_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    metrics = {
        "method": args.method,
        "rank": args.rank,
        "alpha": args.alpha,
        "test_accuracy": best_acc,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "peak_gpu_memory_bytes": peak_mem,
        "wall_clock_seconds": wall_time,
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. best_acc={best_acc:.4f} peak_mem={peak_mem/1e9:.2f}GB wall={wall_time:.1f}s")
    if args.wandb:
        import wandb
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
