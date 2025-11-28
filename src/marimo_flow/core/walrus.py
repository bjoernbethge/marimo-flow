"""Walrus foundation model adapter for PINA."""

from __future__ import annotations

import torch
import torch.nn as nn
from pina import LabelTensor
from transformers import AutoModel


class WalrusAdapter(nn.Module):
    """Wrap the polymathic-ai/walrus checkpoint for use inside PINA."""

    def __init__(
        self,
        checkpoint: str = "polymathic-ai/walrus",
        *,
        out_labels: tuple[str, ...] = ("u",),
        freeze_backbone: bool = True,
        dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, len(out_labels)),
        )
        self.out_labels = out_labels

    def forward(self, coords: LabelTensor) -> LabelTensor:
        """Map geometric coordinates to predicted field values."""

        base = coords.as_tensor()  # shape [N, dim]
        inputs = base.unsqueeze(0)
        outputs = self.backbone(inputs_embeds=inputs)
        features = outputs.last_hidden_state.squeeze(0)
        prediction = self.head(features)
        return LabelTensor(prediction, labels=list(self.out_labels))

