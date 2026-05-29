"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
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
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
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


def apply_lora_to_decoder(decoder: nn.Module, rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """Apply LoRA to q_proj and v_proj in decoder attention layers."""
    from basics.lora import LoRALinear
    import transformers

    for p in decoder.parameters():
        p.requires_grad_(False)

    for module in decoder.modules():
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
            module.q_proj = LoRALinear(module.q_proj, rank, alpha)
        if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
            module.v_proj = LoRALinear(module.v_proj, rank, alpha)

    return decoder


def build_prompt(question: str, injection: str) -> str:
    """Build the text prompt for a CLEVR example."""
    if injection == "interleaved":
        return f"<image> Question: {question} Answer:"
    return f"Question: {question} Answer:"


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    run_name = f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"

    if args.wandb:
        import wandb
        wandb.init(entity="dsheth_caltech", project="cs148b-hw3", name=run_name, config={
            "injection": args.injection, "mask_mode": args.mask_mode,
            "freeze_config": args.freeze_config, **cfg,
        })

    from vlm.data import build_clevr_loaders
    from basics.vit import ViT
    from vlm.projector import VisionLanguageProjector
    from vlm.model import VisionLanguageModel
    from vlm.eval import batch_clevr_accuracy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = torch.load(args.pretrained_vit, map_location="cpu")
    vit_config = ckpt["vit_config"]

    train_dl, val_dl = build_clevr_loaders(
        img_size=vit_config["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    vit = ViT(**vit_config)
    vit.load_state_dict(ckpt["vit_state"])

    decoder_cfg = cfg["decoder"]
    tokenizer = AutoTokenizer.from_pretrained(decoder_cfg["model_name"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id = None
    if args.injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_cfg["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation=decoder_cfg.get("attn_implementation", "eager"),
    )

    if args.injection == "interleaved":
        decoder.resize_token_embeddings(len(tokenizer))

    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=vit_config["d_model"],
        d_decoder=d_decoder,
        expansion=cfg["projector"]["expansion"],
    )

    for p in vit.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    for p in projector.parameters():
        p.requires_grad_(True)

    if args.freeze_config == "B":
        decoder = apply_lora_to_decoder(decoder, rank=8, alpha=16.0)
        for p in decoder.parameters():
            if p.requires_grad:
                pass

    elif args.freeze_config == "C":
        for p in decoder.parameters():
            p.requires_grad_(True)

    elif args.freeze_config == "D":
        for p in vit.parameters():
            p.requires_grad_(True)
        for p in decoder.parameters():
            p.requires_grad_(True)

    vit = vit.to(device)
    projector = projector.to(device)
    decoder = decoder.to(device)

    model = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,}")

    ocfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=ocfg["lr"],
        weight_decay=ocfg["weight_decay"], betas=tuple(ocfg["betas"]),
    )

    num_steps = cfg["train"]["num_steps"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, ocfg["warmup_steps"], num_steps)

    grad_accum = cfg["train"]["gradient_accumulation_steps"]
    log_every = cfg["train"]["log_every"]
    eval_every = cfg["train"]["eval_every_steps"]
    eval_max = cfg["train"]["eval_max_examples"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_acc = 0.0
    global_step = 0
    optimizer.zero_grad()

    data_iter = iter(train_dl)

    while global_step < num_steps:
        model.train()
        accum_loss = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dl)
                batch = next(data_iter)

            images = batch["image"].to(device)
            prompts = [build_prompt(q, args.injection) for q in batch["question"]]
            answers = batch["answer"]

            full_texts = [p + " " + a for p, a in zip(prompts, answers)]
            tokenized = tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)

            prompt_only = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)

            labels = tokenized.input_ids.clone()
            for i in range(labels.shape[0]):
                plen = prompt_only.attention_mask[i].sum().item()
                labels[i, :plen] = -100
            labels[tokenized.input_ids == tokenizer.pad_token_id] = -100

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(
                    images=images,
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    labels=labels,
                    injection=args.injection,
                    mask_mode=args.mask_mode,
                )

            loss = out["loss"] / grad_accum
            loss.backward()
            accum_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        if global_step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step={global_step} loss={accum_loss:.4f} grad_norm={grad_norm:.4f} lr={lr:.6f}")
            if args.wandb:
                import wandb
                wandb.log({
                    "train/loss": accum_loss, "train/grad_norm": grad_norm, "train/lr": lr
                }, step=global_step)

        if global_step % eval_every == 0 or global_step == num_steps:
            model.eval()
            all_preds, all_golds, all_qtypes = [], [], []
            n_eval = 0

            with torch.no_grad():
                for batch in val_dl:
                    if n_eval >= eval_max:
                        break
                    images = batch["image"].to(device)
                    prompts = [build_prompt(q, args.injection) for q in batch["question"]]

                    gen_cfg = cfg.get("generation", {})
                    preds = model.generate(
                        images, prompts, injection=args.injection,
                        max_new_tokens=gen_cfg.get("max_new_tokens", 32),
                        do_sample=gen_cfg.get("do_sample", False),
                    )
                    all_preds.extend(preds)
                    all_golds.extend(batch["answer"])
                    all_qtypes.extend(batch["q_type"])
                    n_eval += len(batch["answer"])

            accs = batch_clevr_accuracy(all_preds, all_golds, all_qtypes)
            peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            print(f"step={global_step} val_acc={accs['overall']:.4f} peak_mem={peak_mem/1e9:.2f}GB")

            if args.wandb:
                import wandb
                log_dict = {f"val/{k}": v for k, v in accs.items()}
                log_dict["val/peak_gpu_memory_gb"] = peak_mem / 1e9
                wandb.log(log_dict, step=global_step)

            if accs["overall"] > best_val_acc:
                best_val_acc = accs["overall"]
                torch.save({
                    "step": global_step,
                    "vit_state": vit.state_dict(),
                    "projector_state": projector.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "tokenizer": tokenizer,
                    "vit_config": vit_config,
                    "injection": args.injection,
                    "mask_mode": args.mask_mode,
                    "freeze_config": args.freeze_config,
                    "val_acc": best_val_acc,
                    "image_token_id": image_token_id,
                }, args.output_dir / "best.pt")
                print(f"  Saved best (val_acc={best_val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
