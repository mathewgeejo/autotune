# Voice Studio

Local voice cloning on Windows: **give it a voice sample, type new text, and generate speech in that voice**. No performance recording, song link, transcript, paid API, or GPU is required.

## Start

1. Run **setup.cmd** once, or rerun it to update an existing installation.
2. Run **launch.cmd**. The app opens at **http://127.0.0.1:7860**.
3. Record or upload **5-15 seconds of clean speech** from one speaker.
4. Type the words in **Text to speak (English)** and click **Generate speech**.
5. Play the result or download the 24 kHz WAV. Edit the text and generate again using the same sample.

Use a sample with the tone and accent you want to reproduce. Music, other speakers, distortion, and room echo reduce similarity. The model approximates voice identity; it cannot guarantee an indistinguishable copy of every voice. This engine generates English speech.

Longer samples can be trimmed with the start/end controls (up to 60 seconds per selection); the app chooses the strongest continuous ten-second section. Text supports up to 5,000 characters. Long text is split at sentence or word boundaries, synthesized in order, and joined into one WAV. A passage preview appears while the remaining text finishes.

## Speed and local storage

The speech pipeline uses quantized **Chatterbox Turbo ONNX** models, two CPU threads, a persistent worker, and cached speaker conditioning. The first generation downloads approximately **0.9 GB** of model files and loads the sessions. Later requests reuse the sessions and cached voice. Cancellation stops the worker and its child processes; the next request loads a fresh worker. Use **Release model from memory** when you want to free its RAM without closing the interface.

After models are downloaded, speech generation works offline. To download ahead of time:

```powershell
.venv/Scripts/python.exe scripts/download_models.py
```

All processing runs on this PC. The app listens on localhost and does not create a public sharing link. Audio samples are never sent to a speech service. Downloads contact Hugging Face only for model files.

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
