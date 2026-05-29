"""§6 — CLIP pretraining with RoPE positional embeddings (1D and 2D ablation).

Usage:
    python scripts/pretrain_clip_rope.py --config configs/clip_eurosat.yaml --rope-type rope1d
    python scripts/pretrain_clip_rope.py --config configs/clip_eurosat.yaml --rope-type rope2d
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--rope-type", choices=["rope1d", "rope2d"], required=True)
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


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path(f"runs/clip_eurosat_{args.rope_type}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    if args.wandb:
        import wandb
        wandb.init(
            entity="dsheth", project="cs148b-hw3",
            name=f"clip_eurosat_{args.rope_type}", config=cfg
        )

    from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES
    from basics.text_encoder import FrozenTextEncoder
    from vlm.clip import ProjectionHeads, init_logit_scale, clip_loss
    from vlm.eval import zeroshot_classification_accuracy

    if args.rope_type == "rope1d":
        from basics.vit_rope import ViTRoPE1D as ViTClass
    else:
        from basics.vit_rope import ViTRoPE2D as ViTClass

    vcfg = cfg["vit"]
    train_dl, val_dl, test_dl = build_eurosat_loaders(
        img_size=vcfg["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    vit = ViTClass(**vcfg).to(device)
    text_enc = FrozenTextEncoder(cfg["text_encoder"]["model_name"]).to(device)

    d_proj = cfg["projection"]["d_proj"]
    proj_heads = ProjectionHeads(vcfg["d_model"], text_enc.embedding_dim, d_proj).to(device)

    class LogitScaleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = init_logit_scale()

    scale_module = LogitScaleModule().to(device)
    logit_scale = scale_module.scale

    ocfg = cfg["optim"]
    params = list(vit.parameters()) + list(proj_heads.parameters()) + list(scale_module.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=ocfg["lr"], weight_decay=ocfg["weight_decay"], betas=tuple(ocfg["betas"]),
    )

    num_epochs = cfg["train"]["num_epochs"]
    total_steps = num_epochs * len(train_dl)
    scheduler = get_cosine_schedule_with_warmup(optimizer, ocfg["warmup_steps"], total_steps)

    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))
    log_every = cfg["train"]["log_every"]
    ln100 = math.log(100)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        vit.train()
        proj_heads.train()
        epoch_loss = 0.0
        for images, captions in train_dl:
            images = images.to(device)
            img_feats = vit(images)
            txt_feats = text_enc(captions).to(device)
            img_proj, txt_proj = proj_heads(img_feats, txt_feats)
            loss = clip_loss(img_proj, txt_proj, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            logit_scale.data.clamp_(max=ln100)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"epoch={epoch+1} step={global_step} loss={loss.item():.4f} lr={lr:.6f}")
                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=global_step)

        val_acc = zeroshot_classification_accuracy(
            vit, proj_heads, text_enc, val_dl, class_prompts, class_indices, device
        )
        avg_loss = epoch_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{num_epochs} avg_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if args.wandb:
            import wandb
            wandb.log({"val/zero_shot_acc": val_acc, "train/epoch_loss": avg_loss}, step=global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch": epoch + 1,
                "vit_state": vit.state_dict(),
                "proj_heads_state": proj_heads.state_dict(),
                "logit_scale": logit_scale.data,
                "val_acc": val_acc,
                "vit_config": vcfg,
                "rope_type": args.rope_type,
            }
            torch.save(ckpt, args.output_dir / "best.pt")
            print(f"  Saved best checkpoint (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
