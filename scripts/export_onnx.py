#!/usr/bin/env python3
"""
Export a trained Conformer checkpoint to ONNX for deployment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TextTokenizer  # noqa: E402
from src.models import ConformerCTCConfig, ConformerCTCModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/base-conformer/best.ckpt"),
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/base-conformer/model.onnx"),
        help="Destination ONNX file.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=400,
        help="Dummy sequence length used during export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(ROOT / "configs/default.yaml")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_cfg_overrides = checkpoint.get("model_config", {})
    vocab_path = ROOT / cfg.data.vocab_path
    tokenizer = TextTokenizer.from_file(vocab_path)

    model_config = ConformerCTCConfig(
        vocab_size=tokenizer.vocab_size,
        input_dim=model_cfg_overrides.get("input_dim", cfg.model.input_dim),
        hidden_dim=model_cfg_overrides.get("hidden_dim", cfg.model.hidden_dim),
        num_layers=model_cfg_overrides.get("num_layers", cfg.model.num_layers),
        num_attention_heads=model_cfg_overrides.get("num_attention_heads", cfg.model.num_attention_heads),
        ff_multiplier=model_cfg_overrides.get("ff_multiplier", cfg.model.ff_multiplier),
        conv_kernel_size=model_cfg_overrides.get("conv_kernel_size", cfg.model.conv_kernel_size),
        dropout=model_cfg_overrides.get("dropout", cfg.model.dropout),
    )
    model = ConformerCTCModel(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_features = torch.randn(1, args.seq_len, model_config.input_dim)
    dummy_lengths = torch.tensor([args.seq_len], dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_features, dummy_lengths),
        args.output,
        input_names=["features", "feature_lengths"],
        output_names=["logits", "output_lengths"],
        dynamic_axes={
            "features": {0: "batch", 1: "time"},
            "feature_lengths": {0: "batch"},
            "logits": {0: "time", 1: "batch"},
            "output_lengths": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
