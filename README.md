# SpeachLTE

Cameroonian Pidgin + Ewondo speech-to-text that still runs on low-end devices. The repo ships a tiny Conformer-CTC model, the data prep it needs, and a demo that actually prints what the model hears.

## Why it exists
Most Cameroonian audio—radio shows, town-hall recordings, WhatsApp notes—mixes Pidgin, French, and local languages like Ewondo. This project gives researchers and product teams a hackable, low-resource ASR stack so they can caption local content, build accessibility tools, or power voice commands without waiting on massive commercial APIs.

## What you actually get
- **Training pipeline** – Hydra-driven script that builds manifests (synthetic by default), extracts mel features with torchaudio, and trains a compact Conformer. Output: `artifacts/base-conformer/best.ckpt`.
- **Data plumbing** – JSONL manifests, text normalization, and tokenizer utilities ready for FLEURS/Common Voice or your own recordings.
- **Demo output** – `scripts/demo_infer.py` loads any checkpoint, runs greedy decoding on a sample clip, and writes the transcript to `outputs/demo_transcript.txt` so you can see tangible words, not just logs.

## Run it (5‑minute tour)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# 1. Train a quick model (uses synthetic audio if you haven't prepared real data yet)
python3.10 scripts/train_model.py training.epochs=1 training.batch_size=2 model.num_layers=2

# 2. Show the transcript produced by that checkpoint
python3.10 scripts/demo_infer.py --checkpoint artifacts/base-conformer/best.ckpt
cat outputs/demo_transcript.txt
```

Need real data? Drop your curated audio + transcripts into `data/raw`, run `scripts/prepare_data.py`, and the same commands train on them.

### Use Common Voice Pidgin (real data)
```bash
export HF_TOKEN=hf_xxx   # Common Voice download requires a Hugging Face token
python3.10 scripts/download_datasets.py --datasets common_voice --language pcm --cv-max-samples 200
python3.10 scripts/prepare_data.py data.languages='[pcm]'
python3.10 scripts/train_model.py data.prepared_manifest=data/manifests/train.jsonl data.val_manifest=data/manifests/val.jsonl training.epochs=3 training.batch_size=4 model.num_layers=4
python3.10 scripts/demo_infer.py --checkpoint artifacts/base-conformer/best.ckpt --manifest data/manifests/test.jsonl
cat outputs/demo_transcript.txt
```
The download step writes real Common Voice clips into `data/raw/pcm`, `scripts/prepare_data.py` builds `train/val/test` manifests, and the final command prints/saves the decoded sentence.

### Evaluate accuracy (WER/CER)
```bash
python3.10 scripts/evaluate_model.py --checkpoint artifacts/base-conformer/best.ckpt --manifest data/manifests/test.jsonl
cat outputs/eval_report.txt
```

### Export to ONNX
```bash
# Requires: pip install onnx onnxscript
python3.10 scripts/export_onnx.py --checkpoint artifacts/base-conformer/best.ckpt --output artifacts/base-conformer/model.onnx
```

## Repo map
- `scripts/train_model.py` – Hydra entrypoint for training and experiment logging.
- `scripts/demo_infer.py` – Greedy-decoding demo with friendly console/file output.
- `src/data/*` – Ingestion, preprocessing, tokenizer, and dataset code.
- `src/models/conformer_ctc.py` – Lightweight Conformer encoder built for CTC.
- `src/training/train.py` – Minimal training harness with AMP and checkpointing.
- `tests/` – Smoke tests for datasets, tokenizers, metrics, and the harness.

See `plan.md` for the roadmap (integration with NeMo, SpeechBrain, ESPnet, Pyannote, Kaldi, etc.) and `docs/tooling.md` for how those ecosystems fit in.
