"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy. Useful for both Problem (vlm_qualitative) and Problem (mrope_impl).

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_prompt(question: str, injection: str) -> str:
    """Build the text prompt for a CLEVR example."""
    if injection == "interleaved":
        return f"<image> Question: {question} Answer:"
    return f"Question: {question} Answer:"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vit_config = ckpt["vit_config"]
    injection = ckpt.get("injection", "all_patches")
    mask_mode = ckpt.get("mask_mode", "causal")
    image_token_id = ckpt.get("image_token_id", None)

    from basics.vit import ViT
    from vlm.projector import VisionLanguageProjector
    from vlm.model import VisionLanguageModel
    from vlm.data import CLEVRMiniDataset, build_clevr_loaders
    from vlm.eval import batch_clevr_accuracy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vit = ViT(**vit_config)
    vit.load_state_dict(ckpt["vit_state"])

    tokenizer = ckpt.get("tokenizer")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    decoder = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        dtype=torch.bfloat16,
    )
    if image_token_id is not None:
        decoder.resize_token_embeddings(len(tokenizer))

    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=vit_config["d_model"],
        d_decoder=d_decoder,
        expansion=4,
    )

    vit.load_state_dict(ckpt["vit_state"])
    projector.load_state_dict(ckpt["projector_state"])
    decoder.load_state_dict(ckpt["decoder_state"])

    vit = vit.to(device)
    projector = projector.to(device)
    decoder = decoder.to(device)

    model = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id)
    model.eval()

    _, val_dl = build_clevr_loaders(
        img_size=vit_config["img_size"], batch_size=32, num_workers=4
    )

    all_preds, all_golds, all_qtypes = [], [], []
    qualitative = []
    n_eval = 0

    with torch.no_grad():
        for batch in val_dl:
            if n_eval >= args.max_eval:
                break
            images = batch["image"].to(device)
            prompts = [build_prompt(q, injection) for q in batch["question"]]
            preds = model.generate(images, prompts, injection=injection, max_new_tokens=32)

            for i, (pred, gold, qtype, question) in enumerate(
                zip(preds, batch["answer"], batch["q_type"], batch["question"])
            ):
                correct = pred.strip().lower() == gold.strip().lower()
                if len(qualitative) < args.num_examples:
                    qualitative.append({
                        "question": question,
                        "gold": gold,
                        "prediction": pred,
                        "correct": correct,
                        "q_type": qtype,
                    })

            all_preds.extend(preds)
            all_golds.extend(batch["answer"])
            all_qtypes.extend(batch["q_type"])
            n_eval += len(batch["answer"])

    accs = batch_clevr_accuracy(all_preds, all_golds, all_qtypes)

    print("\n--- Accuracy Summary ---")
    for k, v in sorted(accs.items()):
        print(f"  {k}: {v:.4f}")

    examples_path = args.output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for ex in qualitative:
            f.write(json.dumps(ex) + "\n")

    summary = {"accuracy": accs, "n_eval": n_eval, "injection": injection, "mask_mode": mask_mode}
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(qualitative)} examples to {examples_path}")
    print(f"Summary saved to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
