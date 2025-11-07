"""
Takes a manifest on disk and turns it into batches of mel features + token ids for the trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from .tokenizer import TextTokenizer


@dataclass(slots=True)
class SpeechDatasetConfig:
    """Configuration for loading audio and extracting mel-spectrogram features."""

    manifest_path: Path
    tokenizer_path: Path
    feature_dim: int = 80  # number of mel filterbanks
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160  # 10ms hop at 16kHz
    win_length: int = 400  # 25ms window at 16kHz
    max_frames: int = 3000  # ~30 seconds at 100 fps
    audio_root: Optional[Path] = None  # root directory for relative audio paths
    # SpecAugment parameters
    apply_spec_augment: bool = True
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2


class SpeechDataset(Dataset):
    """Dataset that loads audio files and extracts mel-spectrogram features."""

    def __init__(self, config: SpeechDatasetConfig, tokenizer: Optional[TextTokenizer] = None, training: bool = True) -> None:
        self.config = config
        self.records = self._load_manifest(config.manifest_path)
        self.tokenizer = tokenizer or TextTokenizer.from_file(config.tokenizer_path)
        self.training = training
        
        # Initialize mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.feature_dim,
            f_min=0.0,
            f_max=config.sample_rate // 2,
        )
        
        # Initialize SpecAugment
        if config.apply_spec_augment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=config.freq_mask_param
            )
            self.time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=config.time_mask_param
            )
        else:
            self.freq_mask = None
            self.time_mask = None

    def _load_manifest(self, manifest_path: Path) -> List[Dict[str, str]]:
        with manifest_path.open("r", encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        record = self.records[index]
        
        # Load and process audio
        audio_path = self._resolve_audio_path(record.get("audio_filepath", ""))
        features = self._load_and_extract_features(audio_path)
        
        # Encode text to tokens
        tokens = torch.tensor(
            self.tokenizer.encode(record.get("text", "")),
            dtype=torch.long,
        )
        
        return {
            "audio_features": features,
            "feature_length": torch.tensor(features.shape[0], dtype=torch.long),
            "tokens": tokens,
            "token_length": torch.tensor(len(tokens), dtype=torch.long),
        }
    
    def _resolve_audio_path(self, audio_filepath: str) -> Path:
        """Resolve audio filepath, handling both absolute and relative paths."""
        audio_path = Path(audio_filepath)
        if not audio_path.is_absolute() and self.config.audio_root:
            audio_path = self.config.audio_root / audio_path
        return audio_path
    
    def _load_and_extract_features(self, audio_path: Path) -> Tensor:
        """Load audio file and extract mel-spectrogram features.
        
        Returns:
            Tensor of shape (time, n_mels) with log mel-spectrogram features
        """
        try:
            # Load audio - torchaudio will auto-detect backend
            import soundfile as sf
            data, sr = sf.read(str(audio_path), dtype='float32')
            waveform = torch.from_numpy(data).unsqueeze(0)  # Add channel dim
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.config.sample_rate
                )
                waveform = resampler(waveform)
            
            # Extract mel-spectrogram
            mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)
            
            # Apply SpecAugment during training
            if self.training and self.config.apply_spec_augment:
                # Apply frequency masking
                for _ in range(self.config.num_freq_masks):
                    mel_spec = self.freq_mask(mel_spec)
                
                # Apply time masking
                for _ in range(self.config.num_time_masks):
                    mel_spec = self.time_mask(mel_spec)
            
            # Convert to log scale
            log_mel = torch.log(mel_spec + 1e-9)
            
            # Transpose to (time, n_mels)
            features = log_mel.squeeze(0).transpose(0, 1)
            
            # Clip to max length
            if features.shape[0] > self.config.max_frames:
                features = features[:self.config.max_frames]
            
            return features
            
        except Exception as e:
            # Fallback to small random tensor on error
            print(f"Warning: Failed to load audio from {audio_path}: {e}")
            return torch.randn(80, self.config.feature_dim)


@dataclass(slots=True)
class BatchItem:
    """Container for collated batch tensors."""

    audio_features: Tensor
    feature_lengths: Tensor
    tokens: Tensor
    token_lengths: Tensor


def speech_collate_fn(batch: Sequence[Dict[str, Tensor]]) -> BatchItem:
    """Collate variable-length batches for CTC training."""
    feature_lengths = torch.stack([item["feature_length"] for item in batch])
    token_lengths = torch.stack([item["token_length"] for item in batch])

    max_frames = int(feature_lengths.max())
    feature_dim = batch[0]["audio_features"].shape[-1]
    padded_features = torch.zeros(len(batch), max_frames, feature_dim, dtype=torch.float32)
    for idx, item in enumerate(batch):
        frames = int(item["feature_length"])
        padded_features[idx, :frames] = item["audio_features"][:frames]

    max_tokens = int(token_lengths.max())
    padded_tokens = torch.full((len(batch), max_tokens), fill_value=0, dtype=torch.long)
    for idx, item in enumerate(batch):
        length = int(item["token_length"])
        if length > 0:
            padded_tokens[idx, :length] = item["tokens"][:length]

    return BatchItem(
        audio_features=padded_features,
        feature_lengths=feature_lengths,
        tokens=padded_tokens,
        token_lengths=token_lengths,
    )
