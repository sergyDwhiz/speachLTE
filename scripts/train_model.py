#!/usr/bin/env python
"""
Hydra keeps the knobs organized; this script actually launches training and tells you what is happening.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

from dataclasses import asdict

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    SpeechDataset,
    SpeechDatasetConfig,
    TextTokenizer,
    speech_collate_fn,
)
from src.models import ConformerCTCConfig, ConformerCTCModel  # noqa: E402
from src.training import TrainerConfig, TrainingHarness  # noqa: E402
from src.utils import configure_logging  # noqa: E402


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _ensure_manifest(manifest_path: Path, raw_root: Path, sample_rate: int) -> None:
    """Create a synthetic manifest (and audio) if none exists."""
    if manifest_path.exists():
        return

    raw_root.mkdir(parents=True, exist_ok=True)
    audio_path = raw_root / "synthetic.wav"
    duration = 1.5  # seconds
    samples = int(duration * sample_rate)
    waveform = (0.02 * np.random.randn(samples)).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate)

    sample_texts = [
        "bonjour na weti we go do today",
        "dis SpeachLTE demo dey hear cameroon pidgin fine",
        "ewondo words fit mix with pidgin for same sentence",
        "we go train small model make e run for bush taxi phones",
    ]
    records = [
        {
            "audio_filepath": str(audio_path),
            "duration": float(duration),
            "text": text,
            "language": "wes",
            "dialect": "wes",
            "speaker_id": f"synthetic_{idx}",
            "split": "train",
        }
        for idx, text in enumerate(sample_texts)
    ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_datasets(
    cfg: DictConfig,
    base_dir: Path,
) -> Tuple[SpeechDataset, SpeechDataset | None, TextTokenizer]:
    vocab_path = _resolve_path(base_dir, cfg.data.vocab_path)
    manifest_path = _resolve_path(base_dir, cfg.data.prepared_manifest)
    raw_root = _resolve_path(base_dir, cfg.data.raw_root)

    _ensure_manifest(manifest_path, raw_root, cfg.data.sample_rate)

    tokenizer = TextTokenizer.from_file(vocab_path)
    dataset_config = SpeechDatasetConfig(
        manifest_path=manifest_path,
        tokenizer_path=vocab_path,
        feature_dim=cfg.model.input_dim,
        sample_rate=cfg.data.sample_rate,
        max_frames=cfg.data.max_frames,
        audio_root=raw_root,
        apply_spec_augment=cfg.data.apply_spec_augment,
    )
    dataset = SpeechDataset(dataset_config, tokenizer=tokenizer, training=True)

    val_dataset: SpeechDataset | None = None
    val_manifest = cfg.data.get("val_manifest", None)
    if val_manifest:
        val_manifest_path = _resolve_path(base_dir, val_manifest)
        if val_manifest_path.exists():
            val_config = SpeechDatasetConfig(
                manifest_path=val_manifest_path,
                tokenizer_path=vocab_path,
                feature_dim=cfg.model.input_dim,
                sample_rate=cfg.data.sample_rate,
                max_frames=cfg.data.max_frames,
                audio_root=raw_root,
                apply_spec_augment=False,
            )
            val_dataset = SpeechDataset(val_config, tokenizer=tokenizer, training=False)

    if not val_dataset and len(dataset) > 1:
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return dataset, val_dataset, tokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    configure_logging()
    print("Training configuration:")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.experiment.seed)
    base_dir = Path(get_original_cwd())

    dataset, val_dataset, tokenizer = _prepare_datasets(cfg, base_dir)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; ensure manifest contains records.")

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=speech_collate_fn,
        num_workers=0,
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=speech_collate_fn,
            num_workers=0,
        )

    model_config = ConformerCTCConfig(
        vocab_size=tokenizer.vocab_size,
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_attention_heads=cfg.model.get("num_attention_heads", 4),
        ff_multiplier=cfg.model.get("ff_multiplier", 4),
        conv_kernel_size=cfg.model.get("conv_kernel_size", 15),
        dropout=cfg.model.dropout,
    )
    model = ConformerCTCModel(model_config)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    output_dir = _resolve_path(base_dir, cfg.experiment.output_dir)
    device_cfg = cfg.training.get("device", "auto")
    if device_cfg == "auto":
        device_cfg = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_config = TrainerConfig(
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        grad_clip=cfg.training.grad_clip,
        device=device_cfg,
        mixed_precision=cfg.training.mixed_precision,
        output_dir=str(output_dir),
    )
    trainer = TrainingHarness(
        model,
        criterion,
        optimizer,
        trainer_config,
        checkpoint_metadata={"model_config": asdict(model_config)},
    )

    print(f"\nStarting training with {len(dataset)} samples")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {trainer.device}")
    print(f"   AMP enabled: {trainer.amp_enabled}")

    trainer.fit(train_loader, val_loader)
    print("\n✅ Training completed! Checkpoints saved to:", output_dir)


if __name__ == "__main__":
    main()
