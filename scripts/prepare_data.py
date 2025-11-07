#!/usr/bin/env python
"""
Collect raw audio, clean it up, and spit out train/val/test manifests people can actually use.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    AudioPreprocessingConfig,
    AudioPreprocessor,
    DatasetIngestionConfig,
    DatasetIngestionPipeline,
    TextNormalizationConfig,
    TextNormalizer,
)


def split_manifest_by_speaker(
    records: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split manifest records by speaker to avoid data leakage.
    
    Args:
        records: List of manifest records
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    random.seed(seed)
    
    # Group records by speaker
    speaker_records = defaultdict(list)
    for record in records:
        speaker_id = record.get("speaker_id", "unknown")
        speaker_records[speaker_id].append(record)
    
    speakers = list(speaker_records.keys())
    random.shuffle(speakers)
    
    if len(speakers) < 3:
        # Fallback to simple record-level split when we do not have enough speakers
        shuffled_records = records[:]
        random.shuffle(shuffled_records)
        total = len(shuffled_records)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        return {
            "train": shuffled_records[:train_end] or shuffled_records[:1],
            "val": shuffled_records[train_end:val_end],
            "test": shuffled_records[val_end:],
        }
    
    n_speakers = len(speakers)
    train_end = int(n_speakers * train_ratio)
    val_end = train_end + int(n_speakers * val_ratio)
    
    train_speakers = speakers[:train_end]
    val_speakers = speakers[train_end:val_end]
    test_speakers = speakers[val_end:]
    
    splits = {"train": [], "val": [], "test": []}
    for speaker in train_speakers:
        splits["train"].extend(speaker_records[speaker])
    for speaker in val_speakers:
        splits["val"].extend(speaker_records[speaker])
    for speaker in test_speakers:
        splits["test"].extend(speaker_records[speaker])
    
    return splits


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Prepare data with ingestion, preprocessing, and splitting."""
    base_dir = Path(get_original_cwd())
    raw_root = Path(cfg.data.raw_root)
    if not raw_root.is_absolute():
        raw_root = base_dir / raw_root
    manifests_dir = Path(cfg.data.manifests_dir)
    if not manifests_dir.is_absolute():
        manifests_dir = base_dir / manifests_dir
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Ingest raw data
    print("Step 1: Ingesting raw audio files...")
    ingestion_config = DatasetIngestionConfig(
        raw_root=raw_root,
        manifest_path=manifests_dir / "raw.jsonl",
        languages=cfg.data.languages,
    )
    ingestion = DatasetIngestionPipeline(ingestion_config)
    manifest_path = ingestion.run()
    print(f"Created raw manifest at {manifest_path}")
    
    # Step 2: Preprocess and normalize
    print("\nStep 2: Preprocessing audio and normalizing text...")
    audio_config = AudioPreprocessingConfig(sample_rate=cfg.data.sample_rate)
    text_config = TextNormalizationConfig()
    preprocessor = AudioPreprocessor(audio_config)
    normalizer = TextNormalizer(text_config)
    
    manifest_records = []
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if line.strip():
                manifest_records.append(json.loads(line))
    
    processed_records = preprocessor.apply(manifest_records)
    normalized = []
    for record in processed_records:
        record["text"] = normalizer.normalize(
            record.get("text", ""),
            record.get("language", "pcm")
        )
        normalized.append(record)
    
    print(f"Processed {len(normalized)} records")
    
    # Step 3: Split into train/val/test
    print("\nStep 3: Splitting data by speaker...")
    splits = split_manifest_by_speaker(
        normalized,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=cfg.experiment.seed,
    )
    
    # Step 4: Write split manifests
    print("\nStep 4: Writing split manifests...")
    for split_name, split_records in splits.items():
        split_path = manifests_dir / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as sink:
            for record in split_records:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_records)} records -> {split_path}")
    
    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
