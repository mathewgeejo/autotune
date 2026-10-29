# Voice Studio

A local Windows audio studio: sing in a reference voice, or turn humming into piano, guitar, violin, and synth. No AI API key or paid service is required.

## Start

1. Run **setup.cmd** once. It installs an isolated Python 3.10 environment, CPU audio packages, FFmpeg, FluidSynth, and a General MIDI soundfont inside this project.
2. Run **launch.cmd**. The interface opens at **http://127.0.0.1:7860**.
3. Record or upload up to 30 seconds of solo singing/humming. For longer files, enter a start and end time.

**Instrument:** choose a sound and click **Transform my voice**. Play or download the WAV, or export MIDI for editing in a music app. Sustained humming with clear note changes works best. Octave shift moves the melody; higher sensitivity picks up quieter notes but may introduce extra notes.

**Singer:** paste a song URL or upload a downloaded song, then click **Prepare song reference**. The app separates vocals, selects a reference section, and previews it. You can adjust the section. Record your singing and click **Transform my voice**. This changes vocal tone while retaining your lyrics, melody, and timing; it does not make you perform the reference song automatically. Similarity varies with recording quality and the model.

Song links use yt-dlp. Site changes, login restrictions, and protected streams can prevent imports; upload the audio file in that case. References must be under 10 minutes and 150 MB. Multiple singers and heavy effects can reduce reference quality.

## Local processing and storage

First use of vocal separation and singer conversion downloads public model weights (several GB). After the weights are cached, file-based processing works offline. URL import always needs internet. The interface uses system fonts and disables Gradio analytics. It listens only on localhost and does not create a public sharing link.

This PC has an Intel i3 CPU and around 8 GB RAM. Processing happens after recording, one job at a time. Singing conversion can take several minutes or longer. The activity panel reports elapsed time and peak worker memory. Cancel stops the worker and its child processes, freeing loaded models. Preview quality uses 10 diffusion steps, Standard 20, and Detailed 30.

All audio is stored under `data/`: `jobs/` holds recordings, results and logs, `references/` holds extracted vocal stems, and `tmp/` holds interface audio. The Local storage controls delete recordings/exports or cached references. Model weights remain in `data/models/` and `vendor/seed-vc/checkpoints/` for reuse. Setup assets live in `vendor/`; the Python runtime and installer cache live in `.tools/`.

## Development and focused checks

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/smoke.py instrument
.venv/Scripts/python.exe scripts/smoke.py singer
.venv/Scripts/python.exe scripts/smoke.py prepare
.venv/Scripts/python.exe app.py --no-browser
```

Smoke checks run actual model inference and save timing/memory reports alongside their job logs. Singer checks use upstream example clips and do not assess similarity to your own recordings. No frontend build is needed. Do not run routine builds after edits.

## Engines and licenses

- [Seed-VC](https://github.com/Plachtaa/seed-vc), GPL-3.0, pinned to `51383efd921027683c89e5348211d93ff12ac2a8`. Setup patches Whisper's forced fp16 dtype to float32 for CPU use; inference uses the upstream singing pipeline with pitch and timing preserved.
- [Demucs](https://github.com/facebookresearch/demucs), MIT, for vocal separation.
- [Basic Pitch](https://github.com/spotify/basic-pitch), Apache-2.0, using ONNX on CPU for audio-to-MIDI.
- [FluidSynth](https://github.com/FluidSynth/fluidsynth), LGPL-2.1, and [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS), with its soundfont license downloaded by setup.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for song imports, and [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) for bundled FFmpeg.

Upstream code and licenses are preserved in the downloaded assets. Check the applicable engine, model-weight, and soundfont licenses before redistributing a packaged app.
