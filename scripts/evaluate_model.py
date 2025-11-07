#!/usr/bin/env python3
"""
Compute WER/CER on a manifest using a trained checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    SpeechDataset,
    SpeechDatasetConfig,
    TextTokenizer,
    speech_collate_fn,
)
from src.evaluation import CharacterErrorRate, WordErrorRate  # noqa: E402
from src.models import ConformerCTCConfig, ConformerCTCModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a manifest.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/base-conformer/best.ckpt"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/test.jsonl"),
        help="Manifest to evaluate on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation dataloader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval_report.txt"),
        help="Where to write the evaluation summary.",
    )
    return parser.parse_args()


def ctc_greedy_decode(ids: List[int], blank_id: int) -> List[int]:
    collapsed: List[int] = []
    prev = None
    for idx in ids:
        if idx == blank_id:
            prev = None
            continue
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return collapsed


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(ROOT / "configs/default.yaml")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    vocab_path = ROOT / cfg.data.vocab_path
    tokenizer = TextTokenizer.from_file(vocab_path)

    dataset_config = SpeechDatasetConfig(
        manifest_path=manifest_path,
        tokenizer_path=vocab_path,
        feature_dim=cfg.model.input_dim,
        sample_rate=cfg.data.sample_rate,
        max_frames=cfg.data.max_frames,
        audio_root=Path(cfg.data.raw_root),
        apply_spec_augment=False,
    )
    dataset = SpeechDataset(dataset_config, tokenizer=tokenizer, training=False)
    if len(dataset) == 0:
        raise ValueError(f"No samples found in manifest {manifest_path}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=speech_collate_fn,
        num_workers=0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg_overrides = checkpoint.get("model_config", {})
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
    model = ConformerCTCModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    wer_metric = WordErrorRate()
    cer_metric = CharacterErrorRate()
    references: List[str] = []
    predictions: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch.audio_features.to(device)
            lengths = batch.feature_lengths
            logits, out_lengths = model(features, lengths)
            # logits: T x B x V
            log_probs = logits.log_softmax(dim=-1)
            preds = log_probs.argmax(dim=-1)  # T x B
            T, B = preds.shape
            for i in range(B):
                seq_len = int(out_lengths[i])
                pred_ids = preds[:seq_len, i].tolist()
                cleaned = ctc_greedy_decode(pred_ids, blank_id=tokenizer.blank_id)
                predicted_text = tokenizer.decode(cleaned)
                reference_text = tokenizer.decode(batch.tokens[i][: batch.token_lengths[i]].tolist())
                predictions.append(predicted_text)
                references.append(reference_text)

    wer_result = wer_metric(references, predictions)
    cer_result = cer_metric(references, predictions)

    summary = (
        f"Samples evaluated: {len(references)}\n"
        f"WER: {wer_result.score:.4f} (distance={wer_result.distance}, words={wer_result.reference_length})\n"
        f"CER: {cer_result.score:.4f} (distance={cer_result.distance}, chars={cer_result.reference_length})\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Manifest: {manifest_path}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
