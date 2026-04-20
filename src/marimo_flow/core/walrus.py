"""Foundation model adapters for PINA."""

from __future__ import annotations

import torch
import torch.nn as nn
from pina import LabelTensor
from transformers import AutoModel


class FoundationModelAdapter(nn.Module):
    """Wrap a Hugging Face backbone for coordinate-to-field prediction."""

    def __init__(
        self,
        checkpoint: str = "polymathic-ai/walrus",
        *,
        input_dimensions: int = 2,
        out_labels: tuple[str, ...] = ("u",),
        freeze_backbone: bool = True,
        dtype: torch.dtype | None = torch.bfloat16,
        model_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if model_kwargs is None:
            model_kwargs = {}

        self.backbone = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.input_projection = nn.Linear(input_dimensions, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, len(out_labels)),
        )
        self.out_labels = out_labels

    def forward(self, coords: LabelTensor) -> LabelTensor:
        """Map geometric coordinates to predicted field values.

        The backbone consumes `inputs_embeds`, so we first project raw coordinates
        into the model hidden size to support arbitrary checkpoints.
        """

        base = torch.as_tensor(coords)  # shape [N, input_dimensions]
        token_embeds = self.input_projection(base).unsqueeze(0)
        outputs = self.backbone(inputs_embeds=token_embeds)
        features = outputs.last_hidden_state.squeeze(0)
        prediction = self.head(features)
        return LabelTensor(prediction, labels=list(self.out_labels))
