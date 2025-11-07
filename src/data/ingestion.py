"""
Dataset ingestion utilities for building manifests from raw audio collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import json
import logging
import torchaudio
import soundfile as sf

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetIngestionConfig:
    """Configuration for scanning raw audio directories and producing manifests."""

    raw_root: Path
    manifest_path: Path
    audio_extensions: Sequence[str] = (".wav", ".flac", ".mp3")
    languages: Sequence[str] = ("wes", "ewo")  # wes ~ Cameroonian Pidgin, ewo ~ Ewondo
    metadata_fields: List[str] = field(
        default_factory=lambda: ["language", "dialect", "speaker_id", "split"]
    )


class DatasetIngestionPipeline:
    """
    Walks the raw dataset directory, collects metadata, and writes JSONL manifests.

    The pipeline expects the following directory structure (adjust as needed):

        raw_root/
            language/
                speaker_id/
                    clip.wav
                    clip.json  # optional metadata sidecar
    """

    def __init__(self, config: DatasetIngestionConfig) -> None:
        self.config = config

    def discover_audio_files(self) -> Iterable[Path]:
        """Yield audio files that match the configured extensions."""
        for ext in self.config.audio_extensions:
            yield from self.config.raw_root.rglob(f"*{ext}")

    def build_record(self, audio_path: Path) -> Dict[str, str]:
        """
        Construct a manifest record for a single audio file.

        Infers basic metadata from folder names and computes actual audio duration.
        """
        parts = audio_path.relative_to(self.config.raw_root).parts
        language = parts[0] if parts else "unknown"
        speaker_id = parts[1] if len(parts) > 1 else "unknown"
        metadata = self._load_sidecar(audio_path)
        
        # Compute actual audio duration
        duration = self._compute_duration(audio_path)
        
        record = {
            "audio_filepath": str(audio_path),
            "duration": duration,
            "text": "",  # Fill with transcription from metadata or preprocessing
            "language": language,
            "dialect": language,
            "speaker_id": speaker_id,
            "split": "unspecified",
        }
        record.update({k: metadata.get(k, record.get(k)) for k in self.config.metadata_fields})
        if "text" in metadata:
            record["text"] = metadata["text"]
        if "duration" in metadata:
            record["duration"] = metadata["duration"]
        return record
    
    def _compute_duration(self, audio_path: Path) -> float:
        """Compute actual audio duration in seconds."""
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            return float(duration)
        except Exception:
            try:
                data, sr = sf.read(str(audio_path))
                return float(len(data) / (sr or 1))
            except Exception as e:
                LOGGER.warning("Failed to compute duration for %s: %s", audio_path, e)
                return -1.0

    def _load_sidecar(self, audio_path: Path) -> Dict[str, str]:
        """Load optional JSON sidecar with the same stem as the audio file."""
        sidecar = audio_path.with_suffix(".json")
        if sidecar.exists():
            try:
                return json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to parse metadata sidecar %s: %s", sidecar, exc)
        return {}

    def run(self) -> Path:
        """Build the manifest file and return its path."""
        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.manifest_path.open("w", encoding="utf-8") as sink:
            for audio_path in self.discover_audio_files():
                record = self.build_record(audio_path)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.config.manifest_path
