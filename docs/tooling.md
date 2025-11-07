# Tooling Integration Notes

This project centers on a custom PyTorch + torchaudio pipeline, but we plan to cross-reference major ASR ecosystems both for benchmarking and collaboration. Below is our working map for each toolkit.

## HuggingFace
- Use `datasets` to download/stream FLEURS, Common Voice, and other community corpora.
- Leverage MMS (Massively Multilingual Speech) checkpoints as teacher models for knowledge distillation.
- Host model cards and evaluation reports on the HuggingFace Hub for community feedback.

## NVIDIA NeMo
- Import pretrained Conformer and Citrinet checkpoints for warm starts.
- Reuse NeMo’s augmentation recipes (SpeedPerturb, NoiseAugment) via YAML-compatible configs.
- Compare our TrainerConfig against NeMo’s Lightning-based training loop to ensure feature parity.

## SpeechBrain
- Benchmark SpeechBrain’s CTC/seq2seq models as baselines on the same manifests.
- Experiment with SpeechBrain’s ECAPA-TDNN speaker embeddings for data curation and augmentation.
- Evaluate SpeechBrain’s hyperparameter tuning utilities against our Hydra sweeps.

## ESPnet
- Adopt ESPnet recipe structure for reproducible experiments and shared logs.
- Investigate ESPnet streaming inference modules to complement our lightweight deployment targets.
- Utilize ESPnet’s joint CTC/Attention decoding strategies for robustness testing.

## Pyannote
- Integrate diarization + VAD for segmenting community recordings before transcription.
- Establish evaluation hooks for diarization error rate and overlap handling in our data pipeline.

## Kaldi
- Reuse Kaldi lexicon/G2P and i-vector extraction for hybrid modeling experiments.
- Convert Kaldi alignments into our manifest format for cross-system comparison.

## Other Tooling
- Maintain the primary code path in raw PyTorch/torchaudio for maximal transparency.
- Export trained weights to ONNX/TFLite for mobile deployment.
- Track experiments with Hydra + (planned) MLflow or Weights & Biases integration.
