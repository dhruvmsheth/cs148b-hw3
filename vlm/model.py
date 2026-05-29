"""Vision-Language Model — §5."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT encoder + projector + pretrained causal LM decoder."""

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def _get_visual_embeds(self, images: torch.Tensor, injection: InjectionMode) -> torch.Tensor:
        """Get projected visual embeddings based on injection mode."""
        if injection == "cls":
            vis_feats = self.vit(images).unsqueeze(1)  # (B, 1, d_image)
        else:
            vis_feats = self.vit(images, return_all_tokens=True)  # (B, N+1, d_image)
        return self.projector(vis_feats)  # (B, n_vis, d_decoder)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        B = images.shape[0]
        vis_embeds = self._get_visual_embeds(images, injection)
        n_visual = vis_embeds.shape[1]

        # Get text token embeddings from decoder
        text_embeds = self.decoder.model.embed_tokens(input_ids)

        if injection == "interleaved":
            # Replace <image> tokens with visual embeddings
            combined_embeds_list = []
            combined_labels_list = []
            combined_mask_list = []
            for i in range(B):
                img_mask = input_ids[i] == self.image_token_id
                img_positions = img_mask.nonzero(as_tuple=True)[0]
                if len(img_positions) > 0:
                    pos = img_positions[0].item()
                    before = text_embeds[i, :pos]
                    after = text_embeds[i, pos + 1:]
                    seq = torch.cat([before, vis_embeds[i], after], dim=0)
                    combined_embeds_list.append(seq)
                    if labels is not None:
                        lbl_before = labels[i, :pos]
                        lbl_after = labels[i, pos + 1:]
                        lbl_img = torch.full((n_visual,), -100, device=labels.device, dtype=labels.dtype)
                        combined_labels_list.append(torch.cat([lbl_before, lbl_img, lbl_after]))
                    mask_before = attention_mask[i, :pos]
                    mask_after = attention_mask[i, pos + 1:]
                    mask_img = torch.ones(n_visual, device=attention_mask.device, dtype=attention_mask.dtype)
                    combined_mask_list.append(torch.cat([mask_before, mask_img, mask_after]))
                else:
                    combined_embeds_list.append(text_embeds[i])
                    if labels is not None:
                        combined_labels_list.append(labels[i])
                    combined_mask_list.append(attention_mask[i])

            # Pad to same length
            max_len = max(e.shape[0] for e in combined_embeds_list)
            d = combined_embeds_list[0].shape[-1]
            inputs_embeds = torch.zeros(B, max_len, d, device=images.device, dtype=combined_embeds_list[0].dtype)
            new_mask = torch.zeros(B, max_len, device=images.device, dtype=attention_mask.dtype)
            new_labels = torch.full((B, max_len), -100, device=images.device, dtype=torch.long) if labels is not None else None

            for i in range(B):
                L = combined_embeds_list[i].shape[0]
                inputs_embeds[i, :L] = combined_embeds_list[i]
                new_mask[i, :L] = combined_mask_list[i]
                if labels is not None:
                    new_labels[i, :L] = combined_labels_list[i]

            attention_mask = new_mask
            labels = new_labels
        else:
            # Prepend visual tokens
            inputs_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
            vis_mask = torch.ones(B, n_visual, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            if labels is not None:
                vis_labels = torch.full((B, n_visual), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([vis_labels, labels], dim=1)

        # Build attention mask for decoder
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        if mask_mode == "image_bidir":
            n_text = inputs_embeds.shape[1] - n_visual
            bidir_mask = build_image_bidir_mask(
                n_visual, n_text, inputs_embeds.device, inputs_embeds.dtype
            )
            kwargs["attention_mask"] = bidir_mask.expand(B, -1, -1, -1)

        if labels is not None:
            kwargs["labels"] = labels

        outputs = self.decoder(**kwargs)
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text conditioned on images + prompts."""
        self.eval()
        vis_embeds = self._get_visual_embeds(images, injection)
        n_visual = vis_embeds.shape[1]

        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(images.device)

        text_embeds = self.decoder.model.embed_tokens(tokenized.input_ids)

        if injection == "interleaved" and self.image_token_id is not None:
            combined_list = []
            for i in range(len(prompts)):
                img_mask = tokenized.input_ids[i] == self.image_token_id
                positions = img_mask.nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    pos = positions[0].item()
                    before = text_embeds[i, :pos]
                    after = text_embeds[i, pos + 1:]
                    combined_list.append(torch.cat([before, vis_embeds[i], after], dim=0))
                else:
                    combined_list.append(text_embeds[i])
            max_len = max(e.shape[0] for e in combined_list)
            d = combined_list[0].shape[-1]
            inputs_embeds = torch.zeros(len(prompts), max_len, d, device=images.device, dtype=text_embeds.dtype)
            attn_mask = torch.zeros(len(prompts), max_len, device=images.device, dtype=torch.long)
            for i, emb in enumerate(combined_list):
                inputs_embeds[i, :emb.shape[0]] = emb
                attn_mask[i, :emb.shape[0]] = 1
        else:
            inputs_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
            B = len(prompts)
            vis_mask = torch.ones(B, n_visual, device=images.device, dtype=torch.long)
            attn_mask = torch.cat([vis_mask, tokenized.attention_mask], dim=1)

        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        # Decode only the newly generated tokens
        generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated
