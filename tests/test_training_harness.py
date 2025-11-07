import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import SpeechDataset, SpeechDatasetConfig, TextTokenizer, speech_collate_fn
from src.models import ConformerCTCConfig, ConformerCTCModel
from src.training import TrainerConfig, TrainingHarness


def _write_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "prepared.jsonl"
    records = [
        {"audio_filepath": "dummy0.wav", "duration": 0.8, "text": "aa"},
        {"audio_filepath": "dummy1.wav", "duration": 0.9, "text": "ab"},
    ]
    with manifest_path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record) + "\n")
    return manifest_path


def _write_vocab(tmp_path: Path) -> Path:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        '{"token_to_id": {"<blank>": 0, "<unk>": 1, "a": 2, "b": 3}}',
        encoding="utf-8",
    )
    return vocab_path


def test_training_harness_runs(tmp_path):
    torch.manual_seed(0)
    manifest_path = _write_manifest(tmp_path)
    vocab_path = _write_vocab(tmp_path)

    tokenizer = TextTokenizer.from_file(vocab_path)
    dataset_config = SpeechDatasetConfig(
        manifest_path=manifest_path,
        tokenizer_path=vocab_path,
        feature_dim=8,
        max_frames=120,
        apply_spec_augment=False,
    )
    dataset = SpeechDataset(dataset_config, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=speech_collate_fn)

    model_config = ConformerCTCConfig(
        input_dim=dataset_config.feature_dim,
        vocab_size=tokenizer.vocab_size,
        hidden_dim=32,
        num_layers=2,
        num_attention_heads=2,
        ff_multiplier=2,
        conv_kernel_size=7,
        dropout=0.1,
    )
    model = ConformerCTCModel(model_config)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer_config = TrainerConfig(
        epochs=1,
        learning_rate=1e-3,
        batch_size=2,
        mixed_precision=False,
        device="cpu",
        output_dir=str(tmp_path / "artifacts"),
        log_every=1,
    )
    trainer = TrainingHarness(model, criterion, optimizer, trainer_config)

    trained_model, _ = trainer.fit(dataloader, None)
    assert isinstance(trained_model, ConformerCTCModel)


def test_training_harness_disables_amp_when_cuda_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_config = ConformerCTCConfig(
        input_dim=8,
        vocab_size=4,
        hidden_dim=16,
        num_layers=2,
        num_attention_heads=2,
        ff_multiplier=2,
        conv_kernel_size=7,
        dropout=0.1,
    )
    model = ConformerCTCModel(model_config)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer_config = TrainerConfig(
        epochs=1,
        learning_rate=1e-3,
        mixed_precision=True,
        device="cuda",
        output_dir=str(tmp_path / "artifacts"),
    )
    trainer = TrainingHarness(model, criterion, optimizer, trainer_config)

    assert trainer.device.type == "cpu"
    assert not trainer.amp_enabled
    assert trainer.scaler is None
