#!/usr/bin/env python3
"""Download LibriSpeech dev-clean subset for quick testing."""

import json
from pathlib import Path
from datasets import load_dataset
import soundfile as sf

def main():
    output_dir = Path("data/raw/librispeech")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading LibriSpeech dev-clean (337 samples, ~20 minutes of speech)...")
    dataset = load_dataset("librispeech_asr", "clean", split="validation[:337]")
    
    manifests = {"train": [], "val": [], "test": []}
    
    for idx, example in enumerate(dataset):
        # Save audio
        audio_path = output_dir / f"sample_{idx:04d}.wav"
        sf.write(audio_path, example["audio"]["array"], example["audio"]["sampling_rate"])
        
        # Create manifest entry
        record = {
            "audio_filepath": str(audio_path.absolute()),
            "text": example["text"].lower(),
            "duration": len(example["audio"]["array"]) / example["audio"]["sampling_rate"],
            "speaker_id": f"speaker_{example['speaker_id']}",
            "language": "en",
            "split": "train" if idx < 270 else ("val" if idx < 303 else "test")
        }
        
        split = record["split"]
        manifests[split].append(record)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/337 samples...")
    
    # Write manifests
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    for split, records in manifests.items():
        manifest_path = manifest_dir / f"{split}_librispeech.jsonl"
        with manifest_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"✅ {split}: {len(records)} samples → {manifest_path}")
    
    print(f"\n✅ Done! Total samples: {len(dataset)}")
    print(f"   Train: {len(manifests['train'])}")
    print(f"   Val: {len(manifests['val'])}")
    print(f"   Test: {len(manifests['test'])}")

if __name__ == "__main__":
    main()
