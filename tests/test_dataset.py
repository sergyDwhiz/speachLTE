import json
from pathlib import Path

import torch

from src.data import SpeechDataset, SpeechDatasetConfig, TextTokenizer


def create_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "prepared.jsonl"
    records = [
        {
            "audio_filepath": str(tmp_path / "clip0.wav"),
            "duration": 1.0,
            "text": "hello",
            "language": "pcm",
        },
        {
            "audio_filepath": str(tmp_path / "clip1.wav"),
            "duration": 0.5,
            "text": "hi",
            "language": "pcm",
        },
    ]
    with manifest_path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record) + "\n")
    return manifest_path


def create_vocab(tmp_path: Path) -> Path:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        '{"token_to_id": {"<blank>": 0, "<unk>": 1, "h": 2, "e": 3, "l": 4, "o": 5, "i": 6}}',
        encoding="utf-8",
    )
    return vocab_path


def test_dataset_shapes(tmp_path):
    manifest_path = create_manifest(tmp_path)
    vocab_path = create_vocab(tmp_path)
    tokenizer = TextTokenizer.from_file(vocab_path)
    config = SpeechDatasetConfig(
        manifest_path=manifest_path,
        tokenizer_path=vocab_path,
        feature_dim=16,
        max_frames=120,
        apply_spec_augment=False,
    )
    dataset = SpeechDataset(config, tokenizer=tokenizer)

    sample = dataset[0]
    assert sample["audio_features"].shape[1] == 16
    assert sample["audio_features"].shape[0] <= config.max_frames
    assert sample["tokens"].dtype == torch.long
