#!/usr/bin/env python3
"""
Download and prepare Cameroonian-focused speech datasets plus the MMS baseline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import login as hf_login


def install_dependencies():
    """Install required packages for dataset download."""
    packages = ["datasets", "huggingface_hub", "tqdm"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def maybe_login(token: Optional[str]) -> None:
    """Authenticate with Hugging Face Hub when a token is provided."""
    token = token or os.getenv("HF_TOKEN")
    if token:
        hf_login(token=token, add_to_git_credential=False, write_permission=True)


def download_fleurs(output_dir: Path, language: str = "ewo_cm"):
    """
    Download FLEURS dataset from HuggingFace.
    
    FLEURS has ~10 hours per language with high-quality transcriptions.
    Ewondo code: 'ewo_cm'
    
    Args:
        output_dir: Directory to save dataset
        language: Language code (ewo_cm for Ewondo; see FLEURS docs)
    """
    from datasets import load_dataset
    
    print(f"\nDownloading FLEURS dataset for {language}...")
    print("This includes ~10 hours of speech with transcriptions")
    
    try:
        # Load FLEURS dataset
        dataset = load_dataset("google/fleurs", language, trust_remote_code=True)
        
        # Create output structure
        lang_dir = output_dir / "fleurs" / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        for split_name, split_data in dataset.items():
            split_dir = lang_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing {split_name} split ({len(split_data)} samples)...")
            
            manifest_path = lang_dir / f"{split_name}.jsonl"
            import json
            
            with open(manifest_path, "w", encoding="utf-8") as f:
                for idx, example in enumerate(split_data):
                    # Save audio file
                    audio_path = split_dir / f"{idx:06d}.wav"
                    
                    # Get audio data
                    audio = example["audio"]
                    array = audio["array"]
                    sr = audio["sampling_rate"]
                    
                    # Save as WAV using scipy or soundfile
                    try:
                        import soundfile as sf
                        sf.write(str(audio_path), array, sr)
                    except ImportError:
                        # Fallback to scipy
                        from scipy.io import wavfile
                        wavfile.write(str(audio_path), sr, array)
                    
                    # Create manifest entry
                    manifest_entry = {
                        "audio_filepath": str(audio_path),
                        "text": example["transcription"],
                        "duration": len(array) / sr,
                        "language": language,
                        "speaker_id": str(example["id"]),
                        "split": split_name,
                    }
                    
                    f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            
            print(f"Saved {len(split_data)} samples to {manifest_path}")
        
        print(f"\nFLEURS dataset downloaded successfully to {lang_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading FLEURS: {e}")
        return False


def download_common_voice(output_dir: Path, language: str = "pcm", max_samples: int = 200):
    """
    Download Mozilla Common Voice (v17) via the datasets library and write manifests.
    """
    from datasets import load_dataset
    import json
    import soundfile as sf

    print(f"\nDownloading Common Voice v17 for {language} (max {max_samples} samples per split)...")
    try:
        splits = ["train", "validation", "test"]
        lang_dir = output_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            try:
                dataset = load_dataset(
                    "mozilla-foundation/common_voice_17_0",
                    language,
                    split=split,
                    trust_remote_code=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  Skipping split '{split}' due to error: {exc}")
                continue

            split_dir = lang_dir / split
            split_dir.mkdir(exist_ok=True)
            manifest_path = lang_dir / f"{split}.jsonl"
            written = 0
            with manifest_path.open("a", encoding="utf-8") as sink:
                for idx, example in enumerate(dataset):
                    if written >= max_samples:
                        break
                    audio = example.get("audio")
                    sentence = example.get("sentence") or ""
                    if not audio or not sentence:
                        continue
                    array = audio["array"]
                    sr = audio["sampling_rate"]
                    audio_path = split_dir / f"{idx:06d}.wav"
                    sf.write(str(audio_path), array, sr)
                    entry = {
                        "audio_filepath": str(audio_path),
                        "text": sentence,
                        "duration": len(array) / sr,
                        "language": language,
                        "speaker_id": str(example.get("client_id", "unknown")),
                        "split": split,
                    }
                    sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    written += 1
            print(f"  {split}: wrote {written} samples to {manifest_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Error downloading Common Voice: {exc}")
        return False


def download_mms_model():
    """
    Download Meta's MMS (Massively Multilingual Speech) pretrained model.
    
    This can be used as:
    1. Feature extractor for your model
    2. Starting point for fine-tuning
    3. Baseline comparison
    """
    print("\nDownloading Meta MMS pretrained model...")
    print("Model: facebook/mms-1b-all (supports 1000+ languages)")
    
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        model_name = "facebook/mms-1b-all"
        cache_dir = Path("artifacts/pretrained/mms")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("Downloading processor...")
        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=str(cache_dir))
        
        print("Downloading model...")
        model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=str(cache_dir))
        
        print(f"MMS model downloaded to {cache_dir}")
        print("\nYou can now:")
        print("1. Use it for zero-shot inference on your data")
        print("2. Fine-tune it on your Cameroonian Pidgin/Ewondo dataset")
        return True
        
    except Exception as e:
        print(f"Error downloading MMS model: {e}")
        print("Install transformers: pip install transformers")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Cameroonian language speech datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["fleurs", "common_voice", "mms"],
        default=["fleurs"],
        help="Which datasets to download",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pcm",
        help="Language code (pcm for Common Voice Pidgin, fr for French, etc.)",
    )
    parser.add_argument(
        "--cv-max-samples",
        type=int,
        default=200,
        help="Maximum samples per split when downloading Common Voice.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token (or set HF_TOKEN env var).",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cameroonian Language ASR Dataset Downloader")
    print("=" * 60)
    
    # Install dependencies and authenticate if needed
    print("\nChecking dependencies...")
    install_dependencies()
    maybe_login(args.hf_token)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Download requested datasets
    if "fleurs" in args.datasets:
        results["fleurs"] = download_fleurs(args.output_dir, args.language)
    
    if "common_voice" in args.datasets:
        cv_lang = args.language.split("_")[0]
        results["common_voice"] = download_common_voice(
            args.output_dir,
            cv_lang,
            max_samples=args.cv_max_samples,
        )
    
    if "mms" in args.datasets:
        results["mms"] = download_mms_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for dataset, success in results.items():
        status = "ok" if success else "failed"
        print(f"{dataset}: {status}")
    
    print("\nNext steps:")
    print("1. Check downloaded data in:", args.output_dir)
    print("2. Run data preparation:\n     python scripts/prepare_data.py")
    print("3. Start training:\n     python scripts/train_model.py")


if __name__ == "__main__":
    main()
