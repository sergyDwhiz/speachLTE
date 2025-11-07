"""
Audio preprocessing, augmentation, and text normalization components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch import Tensor

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - soundfile required for real audio pipelines.
    sf = None  # type: ignore[assignment]


@dataclass(slots=True)
class AudioPreprocessingConfig:
    """Parameters controlling audio resampling and augmentation."""

    sample_rate: int = 16000
    target_channels: int = 1
    apply_vad: bool = True
    augment_noise: bool = True
    noise_snr_db: List[int] = field(default_factory=lambda: [5, 10, 15])
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])


@dataclass(slots=True)
class TextNormalizationConfig:
    """Text normalization rules for Cameroonian Pidgin and Ewondo."""

    lowercase: bool = True
    strip_punctuation: bool = True
    preserve_tone_marks: bool = True
    expand_abbreviations: bool = True
    whitelist_chars: Optional[str] = None


class AudioPreprocessor:
    """Audio preprocessing pipeline with loading, resampling, and augmentation."""

    def __init__(self, config: AudioPreprocessingConfig) -> None:
        self.config = config

    def load_audio(self, path: Path) -> Tuple[Tensor, int, float]:
        """
        Load audio file and return waveform, sample rate, and duration.
        
        Returns:
            Tuple of (waveform, sample_rate, duration_seconds)
        """
        try:
            waveform, sample_rate = torchaudio.load(str(path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.sample_rate
                )
                waveform = resampler(waveform)
                sample_rate = self.config.sample_rate
            
            duration = waveform.shape[1] / sample_rate
            return waveform, sample_rate, duration
            
        except Exception as e:
            print(f"Warning: Failed to load audio from {path}: {e}")
            # Return silence as fallback
            return torch.zeros(1, self.config.sample_rate), self.config.sample_rate, 1.0
    
    def get_audio_info(self, path: Path) -> Dict[str, float]:
        """Get audio metadata (duration, sample rate) without loading full waveform."""
        try:
            info = torchaudio.info(str(path))
            duration = info.num_frames / info.sample_rate
            return {
                "path": str(path),
                "sample_rate": info.sample_rate,
                "duration": float(duration),
            }
        except Exception as e:
            if sf is not None:
                try:
                    data, sr = sf.read(str(path))
                    return {
                        "path": str(path),
                        "sample_rate": sr,
                        "duration": float(len(data) / (sr or 1)),
                    }
                except Exception as sf_err:
                    print(f"Warning: Failed to get info from {path}: {sf_err}")
            else:
                print(f"Warning: Failed to get info from {path}: {e}")
            return {"path": str(path), "sample_rate": self.config.sample_rate, "duration": -1.0}
    
    def apply_speed_perturbation(self, waveform: Tensor, speed_factor: float) -> Tensor:
        """Apply speed perturbation to waveform."""
        if speed_factor == 1.0:
            return waveform
        
        effects = [["speed", str(speed_factor)], ["rate", str(self.config.sample_rate)]]
        perturbed, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.config.sample_rate, effects
        )
        return perturbed
    
    def apply_noise_injection(self, waveform: Tensor, snr_db: float) -> Tensor:
        """Add background noise at specified SNR."""
        # Generate white noise
        noise = torch.randn_like(waveform)
        
        # Calculate signal and noise power
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        # Calculate scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        return waveform + scale * noise

    def apply(self, manifest: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply preprocessing to manifest records and compute durations."""
        processed_records: List[Dict[str, str]] = []
        for record in manifest:
            audio_info = self.get_audio_info(Path(record["audio_filepath"]))
            record["duration"] = str(audio_info["duration"])
            processed_records.append(record)
        return processed_records


class TextNormalizer:
    """Normalize transcriptions prior to tokenization."""

    def __init__(self, config: TextNormalizationConfig) -> None:
        self.config = config

    def normalize(self, text: str, language: str) -> str:
        """Apply lightweight rules; extend with language-specific logic."""
        normalized = text
        if self.config.lowercase:
            normalized = normalized.lower()
        if self.config.strip_punctuation:
            normalized = "".join(char for char in normalized if char.isalnum() or char.isspace())
        # TODO: integrate language-specific rules for tone marks and Pidgin orthography.
        return normalized.strip()
