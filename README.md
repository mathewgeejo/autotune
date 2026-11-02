# Voice Studio

Self-hosted voice cloning and audio diagnostics for **English, Hindi, and Malayalam**. Upload or record a sample, type new words, generate speech, and compare the result with the reference. Inference runs on your server; phones and tablets use the browser interface. No paid speech API is configured.

**Quality status:** English generation and the new multilingual pipeline run locally. Indic cloning remains experimental: real tests found pronunciation and speaker-similarity failures, especially with a reference in another language. Near-indistinguishable quality has **not** been established. See [VERIFICATION.md](VERIFICATION.md) for failures as well as passing checks. Cloud deployment is still pending and must be the final step.

## Start

1. Run **setup.cmd** once, or rerun it to update an existing installation.
2. Run **launch.cmd**. The app opens at **http://127.0.0.1:7860**.
3. Record or upload **up to one minute** of clean speech from one speaker. Five to ten clear seconds are often sufficient.
4. Type the words in **Text to speak**, choose Auto or the language, and click **Generate speech**. Use native Hindi or Malayalam script for those languages.
5. Play the result or download the 24 kHz WAV. Edit the text and generate again using the same sample.

Use a sample with the tone and accent you want to reproduce. Music, other speakers, distortion, and room echo reduce similarity. English uses Chatterbox Turbo ONNX. Hindi and Malayalam use a separate public DhVaani ONNX export. Detailed uses full precision, 32 refinement steps, and stronger voice guidance; Balanced uses smaller INT8 graphs and 16 steps. Indic generation can take several minutes for a short sentence on a two-core laptop.

For Hindi/Malayalam, open **Selected reference and transcript** to inspect the selected audio and its words. Blank reference words trigger local recognition. A reference in the output language generally gives more reliable timing. A Malayalam-specific recognizer is used for Malayalam; recognition can still be wrong. The app rejects an incompatible transcript script instead of silently using it.

Longer samples can be trimmed with the start/end controls (up to 60 seconds per selection); the app chooses the strongest continuous ten-second section. Text supports up to 5,000 characters. Long text is split at sentence or word boundaries, synthesized in order, and joined into one WAV. A passage preview appears while the remaining text finishes.

## Speed and local storage

The speech pipeline uses quantized **Chatterbox Turbo ONNX** models, two CPU threads, a persistent worker, and cached speaker conditioning. The first generation downloads approximately **0.9 GB** of model files and loads the sessions. Later requests reuse the sessions and cached voice. Cancellation stops the worker and its child processes; the next request loads a fresh worker. Use **Release model from memory** when you want to free its RAM without closing the interface.

After the selected models are downloaded, file-based processing works offline. Download English ahead of time, or cache every engine and analysis model:

```powershell
.venv/Scripts/python.exe scripts/download_models.py
.venv/Scripts/python.exe scripts/download_models.py --all
```

The complete model set needs approximately 6.5 GB of storage, plus dependencies and recordings. Individual options `--language ml --quality Detailed` and `--audit` avoid downloading everything. Set `HF_HUB_OFFLINE=1` after caching to forbid model-network lookups.

Malayalam/Hindi Detailed mode can optionally use an Intel GPU through `requirements-intel.txt`. Install that optional runtime with `.venv/Scripts/python.exe -m pip install -r requirements-intel.txt`. `VOICE_STUDIO_ACCELERATOR=auto` (the default) selects a compatible Intel GPU when the package is installed; `cpu` forces the existing CPU path. Acceleration retains the FP32 model and 32 steps, caches compilation, and falls back to CPU if initialization or a refinement step fails. English and Balanced mode retain their existing engines. First-use compilation takes extra time, and performance depends on the device. This optional path is undergoing broader voice-quality checks; see [QUALITY_RESULTS.md](QUALITY_RESULTS.md).

All audio processing runs on the host. The app listens on localhost by default and does not create a public sharing link. Audio samples are never sent to a speech service. Downloads contact Hugging Face for model files.

## Compare and debug

Use **Voice analysis** to upload two clips or copy your latest generated result. Acoustic charts measure pitch/voicing, loudness, energy, clipping, silence, and spectra. Enable **Both clips contain the same spoken words** only when comparing readings of the same passage; this adds time alignment and localized differences. Full quality audit adds an independent speaker verifier and multilingual transcription. Expected words are used for scoring only, never as a recognition prompt.

Export the measurements as JSON, frame-level CSV, and a ZIP containing diagnostics. Speaker cosine is a model score, not a percentage of identity. Recognition errors are not automatically synthesis errors; incompatible-script transcripts are withheld from content scoring. Loudness and level are measured on decoded 48 kHz mono; pitch, frame energy, and spectra use a 16 kHz speech-band signal. There is no true-peak or perceptual identity guarantee.

## Other devices and Docker

See [SELF_HOSTING.md](SELF_HOSTING.md) for Linux, Docker Compose, persistent storage, instance login, network binding, and mobile HTTPS requirements. The Docker files are prepared; this Windows workstation has no Docker/WSL runtime, so container execution remains unverified. No cloud resource has been deployed or billed.

`data/jobs/` contains owned sample copies, the selected reference, generated speech, passage previews, logs, and timing reports. `data/references/` contains cached speaker conditioning. `data/models/` holds reusable model weights. Gradio temporary audio lives in `data/tmp/`. The local-storage controls delete recordings/exports or cached voice data while preserving models. Existing singing/instrument models from an older installation are left on disk but are no longer loaded or downloaded by this app.

## Development and checks

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/smoke.py --reference path/to/voice.wav --repeat 2
.venv/Scripts/python.exe app.py --no-browser --port 7861
.venv/Scripts/python.exe scripts/ui_smoke.py --port 7861 --reference path/to/voice.wav
```

Browser checks use installed Chrome and an optional Playwright install under `.tools/browser` (install using the same Python version as the app). `scripts/microphone_smoke.py` uses a simulated microphone with `data/smoke/dave.wav`; it never accesses the physical microphone. See [VERIFICATION.md](VERIFICATION.md) for measured results and limits. No frontend build is needed; the application runs directly with Python.

Setup uses `requirements.lock.txt`; direct dependencies are in `requirements.txt`. The runtime no longer installs Torch, singing conversion, source separation, MIDI transcription, or instrument rendering.

## Engine provenance

The [Chatterbox Turbo ONNX models](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) are published by Resemble AI under MIT. This app pins revision `d21799bd0354adb85e348b8a0442a8405110a2cf`, uses the publisher's 4-bit language model and 8-bit encoder/embedding/decoder graphs, and follows its reference-conditioned inference procedure. ONNX Runtime, Hugging Face Hub, Tokenizers, Gradio, and bundled FFmpeg retain their own upstream licenses. Preserve applicable licenses when redistributing the app or models.

See [THIRD_PARTY.md](THIRD_PARTY.md) for the DhVaani/Vocos Indic engine, independent speaker verifier, and multilingual/Malayalam recognizers. Model revisions and tokenizer assets are pinned in `studio/models.py`; the Indic ONNX export is a third-party conversion.
