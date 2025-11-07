#!/usr/bin/env python
"""
Run a friendly inference demo: load a checkpoint, decode one utterance, and save the transcript we got.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import SpeechDataset, SpeechDatasetConfig, TextTokenizer  # noqa: E402
from src.models import ConformerCTCConfig, ConformerCTCModel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy decoding demo for SpeachLTE.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/base-conformer/best.ckpt"),
        help="Path to a trained checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/prepared.jsonl"),
        help="Manifest file to draw an utterance from.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("configs/vocab.json"),
        help="Character vocabulary JSON.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Which row from the manifest to decode.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demo_transcript.txt"),
        help="Where to save the reference/prediction text.",
    )
    return parser.parse_args()


def _ensure_manifest(manifest_path: Path, raw_root: Path, sample_rate: int) -> None:
    """Create a lightweight synthetic manifest/audio clip if the user has none yet."""
    if manifest_path.exists():
        return

    raw_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = manifest_path.parent
    manifests_dir.mkdir(parents=True, exist_ok=True)

    audio_path = raw_root / "synthetic.wav"
    duration = 1.5
    samples = int(duration * sample_rate)
    waveform = (0.02 * np.random.randn(samples)).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate)

    sample_texts = [
        "bonjour na weti we go do today",
        "dis SpeachLTE demo dey hear cameroon pidgin fine",
        "ewondo words fit mix with pidgin for same sentence",
        "we go train small model make e run for bush taxi phones",
    ]
    with manifest_path.open("w", encoding="utf-8") as sink:
        for idx, text in enumerate(sample_texts):
            record = {
                "audio_filepath": str(audio_path),
                "duration": float(duration),
                "text": text,
                "language": "pcm",
                "dialect": "pcm",
                "speaker_id": f"synthetic_{idx}",
                "split": "demo",
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_dataset(
    manifest_path: Path,
    vocab_path: Path,
    audio_root: Path,
    sample_rate: int,
    feature_dim: int,
    max_frames: int,
) -> Tuple[SpeechDataset, TextTokenizer]:
    tokenizer = TextTokenizer.from_file(vocab_path)
    config = SpeechDatasetConfig(
        manifest_path=manifest_path,
        tokenizer_path=vocab_path,
        feature_dim=feature_dim,
        sample_rate=sample_rate,
        max_frames=max_frames,
        audio_root=audio_root,
        apply_spec_augment=False,
    )
    dataset = SpeechDataset(config, tokenizer=tokenizer, training=False)
    if len(dataset) == 0:
        raise ValueError(f"The manifest at {manifest_path} had no samples to decode.")
    return dataset, tokenizer


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load("configs/default.yaml")

    _ensure_manifest(args.manifest, Path(cfg.data.raw_root), cfg.data.sample_rate)
    dataset, tokenizer = _load_dataset(
        manifest_path=args.manifest,
        vocab_path=args.vocab,
        audio_root=Path(cfg.data.raw_root),
        sample_rate=cfg.data.sample_rate,
        feature_dim=cfg.model.input_dim,
        max_frames=cfg.data.max_frames,
    )

    sample_index = max(0, min(args.sample_index, len(dataset) - 1))
    sample = dataset[sample_index]

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise KeyError(f"{args.checkpoint} did not contain a 'model_state' entry.")
    config_override = checkpoint.get("model_config", {})
    model_config = ConformerCTCConfig(
        vocab_size=tokenizer.vocab_size,
        input_dim=config_override.get("input_dim", cfg.model.input_dim),
        hidden_dim=config_override.get("hidden_dim", cfg.model.hidden_dim),
        num_layers=config_override.get("num_layers", cfg.model.num_layers),
        num_attention_heads=config_override.get("num_attention_heads", cfg.model.num_attention_heads),
        ff_multiplier=config_override.get("ff_multiplier", cfg.model.ff_multiplier),
        conv_kernel_size=config_override.get("conv_kernel_size", cfg.model.conv_kernel_size),
        dropout=config_override.get("dropout", cfg.model.dropout),
    )
    model = ConformerCTCModel(model_config)
    model.load_state_dict(state_dict)
    model.eval()

    features = sample["audio_features"].unsqueeze(0)
    lengths = sample["feature_length"].unsqueeze(0)
    with torch.no_grad():
        logits, out_lengths = model(features, lengths)
        pred_ids = logits.argmax(dim=-1)[0][: out_lengths[0]].tolist()

    reference_text = tokenizer.decode(sample["tokens"].tolist())
    predicted_text = tokenizer.decode(pred_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"Reference: {reference_text}\nPrediction: {predicted_text or '[blank]'}\n",
        encoding="utf-8",
    )

    print("\nDemo complete!")
    print(f"Reference : {reference_text}")
    print(f"Prediction: {predicted_text or '[blank]'}")
    print(f"\nTranscript saved to {args.output}")


if __name__ == "__main__":
    main()
