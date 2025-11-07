"""
A compact Conformer encoder that plays nicely with CTC while staying mobile-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class ConformerCTCConfig:
    """Configuration for the lightweight Conformer encoder."""

    input_dim: int = 80
    vocab_size: int = 30
    hidden_dim: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    ff_multiplier: int = 4
    conv_kernel_size: int = 15
    dropout: float = 0.1


class FeedForwardModule(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, hidden_dim: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        inner_dim = hidden_dim * multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Depthwise separable convolutional module."""

    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pointwise_conv1 = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.pointwise_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: batch x time x hidden_dim
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # batch x hidden_dim x time
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Single Conformer encoder block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_multiplier: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ff_scale = 0.5
        self.ff_module1 = FeedForwardModule(hidden_dim, ff_multiplier, dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(hidden_dim, conv_kernel_size, dropout)
        self.ff_module2 = FeedForwardModule(hidden_dim, ff_multiplier, dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        residual = x
        x = residual + self.ff_scale * self.ff_module1(x)

        attn_input = self.self_attn_norm(x)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)

        x = x + self.conv_module(x)
        x = x + self.ff_scale * self.ff_module2(x)
        return self.final_norm(x)


class ConformerCTCModel(nn.Module):
    """Lightweight Conformer encoder with CTC projection head."""

    def __init__(self, config: ConformerCTCConfig) -> None:
        super().__init__()
        self.config = config
        self.prenet = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_attention_heads,
                    ff_multiplier=config.ff_multiplier,
                    conv_kernel_size=config.conv_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.projection = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, features: Tensor, feature_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: batch x time x feature_dim tensor.
            feature_lengths: batch tensor with sequence lengths.
        Returns:
            logits: time x batch x vocab tensor for CTC.
            output_lengths: batch tensor after potential subsampling (identity here).
        """
        x = self.prenet(features)
        feature_lengths = feature_lengths.to(x.device)
        max_len = x.size(1)
        device = x.device
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= feature_lengths.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)

        x = self.output_norm(x)
        logits = self.projection(x)
        return logits.transpose(0, 1), feature_lengths.cpu()

    def export(self) -> nn.Module:
        """Return a TorchScript-friendly version of the encoder."""
        return self.eval()
