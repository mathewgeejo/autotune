# Voice-cloning verification - 2 November 2026

## Current multilingual work

The expanded pipeline is **not verified as near-indistinguishable**. Tests now exercise real Hindi/Malayalam generation and independent recognizers/verifiers, alongside the existing English path. Bad outputs and recognizer failures are retained in the local reports.

| Check | Current evidence |
| --- | --- |
| Regression suite | **20 passed** after adding script validation and full-band loudness coverage. No application build was run. |
| Minute input boundary | Actual 60-second stereo speech accepted as WAV, FLAC, MP3, and AAC/M4A. Selected conditioning window was ten seconds. A 61-second recording was rejected. Report: `data/smoke/minute-inputs/checks.json`. |
| Updated generation UI | Two actual English generations via Chrome passed with WAV links and cached voice reuse; simulated microphone capture also generated speech without JavaScript errors. |
| Acoustic comparison UI | Actual uploads, six chart traces, JSON/CSV/ZIP downloads, and widths 1440/768/390/360 passed. Layout tests simulate viewports; they are not tests on physical phones. |
| Accessibility | Axe-core WCAG 2.1 A/AA scan reports zero violations on Create speech and Voice analysis at 1440 and 390 pixels. Keyboard focus proceeds from text to Generate, with a visible outline. Report: `data/smoke/accessibility.json`. This automated scan does not establish complete screen-reader accessibility. |
| Clean dependency environment | Locked 85-package runtime imported FasterWhisper and the UI without Torch. Setuptools is pinned because CTranslate2 still imports `pkg_resources`. The main environment was synced to the lean lock, removing obsolete Torch/music and experimental OpenVINO packages. |
| Process portability | Windows tree cancellation passes. POSIX process groups and Tini are configured, but actual Linux/container execution has not been tested on this host. |
| Hosting | Docker/Compose files and documentation prepared. No Docker/WSL runtime installed; no cloud deployment or paid resource created. |

### Quality failures and evaluator calibration

- Balanced INT8 with an English reference produced a Malayalam clip that the recognizer heard as English. A high speaker score on that output did **not** imply correct Malayalam content. Evidence: `multilingual-balanced.json` and `quality-benchmark-v0-raw-levels.json` under `data/smoke`.
- Detailed FP32 with a natural Malayalam reference produced intelligible native-script output, but the original generic recognizer missed word endings. The Hindi test also had a recognized word difference. These are review findings, not a quality pass.
- The generic large-v3-turbo recognizer rendered even a **natural Malayalam recording** in mixed Gujarati/Punjabi scripts. Its resulting word-error score is not a valid measurement of the synthesizer. The app now validates transcript script and withholds incompatible content scores. A pinned native Malayalam recognizer was added for calibration.
- WavLM same-human/different-word controls varied widely with sample (approximately 0.61-0.96 after level matching). There is no universal 0.8 identity threshold; automated speaker cosine must be interpreted against controls, recording conditions, and listening. A review target in the benchmark is not a perceptual pass criterion.
- Experimental OpenVoice tone conversion reduced the independent Malayalam speaker score in the measured trial (approximately 0.368 to 0.312), without improving its recognized words. It was rejected and removed from runtime code. The negative result is retained in `data/smoke/refinement-ml/audit/quality-audit.json`.
- An Intel integrated-GPU OpenVINO experiment made warm diffusion steps faster but added about 79 seconds of compilation. It was not enabled in production or added to the lock.
- One long test crossed Windows Modern Standby. Logged computation was about 278 seconds while elapsed wall time included several hours asleep. That run was canceled and is excluded from latency claims. Active-work timers now exclude Windows sleep, and work-scoped power requests inhibit automatic idle sleep while allowing the screen to turn off.

The sampled detailed Indic generations took roughly 204-218 seconds for 2.7-3.3 seconds of output on this laptop. Download time, background load, and suspend behavior must be separated from inference time. Blind human listening, stronger native-language quality, a real Linux container run, and a verified zero-cost cloud allocation remain open gates in `WORK_PLAN.md`.

## Historical English baseline

The following measurements predate the multilingual/analysis expansion. In particular, the speaker scores below use the synthesizer's own encoder and the old 11-test count is not the current regression count.

Tested on Windows, Intel Core i3-1115G4 (two physical cores), 7.7 GiB RAM, no CUDA GPU. These results describe the new sample-to-text speech pipeline; the old singing and instrument checks do not establish voice-cloning behavior.

| Requirement | Current evidence |
| --- | --- |
| Voice sample + typed text -> generated speech | Real Chatterbox Turbo ONNX synthesis with two different public speech samples (Dave and Jo from the [NeuTTS sample directory](https://github.com/neuphonic/neutts/tree/main/samples)). New words are synthesized using reference conditioning, with no performance recording or reference transcript. |
| Spoken text accuracy | Local Whisper-small recognized every word of the 4.28-second test output, ignoring punctuation/case: "Hello there. This is a new sentence, spoken using the voice in the sample." Expected and recognized text are recorded in `data/smoke/transcript-check.json`. The full two-passage output also matches every word after normalizing punctuation/case and 9/nine, including the final words "see you soon" (`data/smoke/long-text-report.json`). |
| Speaker identity changes with the reference | Quantized speaker-encoder cosine similarity: Dave reference/output **0.870**, Jo reference/output **0.916**. Cross-speaker comparisons were lower (**0.561** and **0.475**). Evidence: `data/smoke/voice-similarity.json`. These are model-based similarity scores, not percentages or proof of an indistinguishable human voice. |
| Faster CPU generation | Same machine, same single-thread language-model benchmark: 8-bit graph took approximately **3.1 s per decoding step**, 4-bit graph approximately **0.03-0.08 s**. The app selects the 4-bit language model with two threads, while preserving 8-bit reference encoding, embeddings, and waveform decoder. |
| Cold/warm runtime | A 4.28-second WAV took **21.9 s cold** (models cached, sessions loaded on demand) and **12.7 s warm**. Same worker PID and cached voice reused. Peak sampled worker RSS approximately **882 / 924 MB**. Evidence: `data/smoke/cloning-report.json` and corresponding job metrics. Model-download time is excluded. |
| Browser upload and export | Chrome uploaded Dave and Jo samples, submitted new text twice, displayed generated waveforms and WAV download links, and passed desktop/mobile layout checks. The final rerun had no JavaScript errors. Screenshots: `data/smoke/cloning-result.png`, `cloning-desktop.png`, `cloning-mobile.png`. |
| Long text and early preview | Actual two-passage generation produced **16.48 s** of speech in **74.6 s** in the browser, approximately **1,203 MB** peak worker RSS. The first passage was playable while the second was generated. The final WAV contains exactly the sum of both passage frame counts, in order (**395,520 frames**). Evidence: `data/smoke/long-text-report.json`. |
| Microphone flow | Chrome recorded a simulated microphone, submitted typed text, and generated **3.12 s** of speech in **17.6 s**, without JavaScript errors. No physical microphone was accessed. Screenshot: `data/smoke/studio-microphone.png`; job `3b703993f5a3458fbe4feeb94d0333a6`. |
| Cancellation and recovery | Cancelled real active synthesis, confirmed every worker-tree PID was gone, then generated real speech in a fresh worker. This ran offline in a clean environment with no Torch installed. Evidence: `data/smoke/lifecycle-report.json`. |
| Offline use | Browser synthesis and clean-runtime cancellation/recovery ran with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Required model files were loaded from the project cache. |
| Fresh installation dependencies | Created `.tools/runtime-check` from the locked **75-package** requirements, with **no Torch installed**. The interface loaded, real speech synthesis completed, and dependency checks passed. |
| Focused tests | **11 passed** in the clean environment: audio decode/selection; silence, corrupt data and reference limits; long-text preservation and limits; last speech token preservation; process-tree cancellation; storage deletion boundaries; first and repeated upload handling. |
| Static/setup checks | Ruff correctness checks pass. PowerShell setup parses and bundled FFmpeg asset validation succeeds. No application builds were run. |

## Reproduce

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/download_models.py
.venv/Scripts/python.exe scripts/smoke.py --reference path/to/speech.wav --repeat 2
.venv/Scripts/python.exe app.py --no-browser --port 7861
.venv/Scripts/python.exe scripts/ui_smoke.py --port 7861 --reference path/to/speech.wav --long-text
.venv/Scripts/python.exe scripts/microphone_smoke.py
```

The optional browser scripts need Playwright installed under `.tools/browser` using the app's Python version and an installed Chrome browser. The microphone check additionally uses `data/smoke/dave.wav` as its simulated capture source. Local reports/audio are intentionally ignored by Git.

## Practical limits

- This historical baseline covers English voice cloning. The expanded app now also routes Hindi/Malayalam to an Indic engine. No engine can guarantee an exact, indistinguishable copy for every reference. Tone, accent, speaking style, noise, and recording quality affect the result; human listening assessment remains necessary.
- Speed varies with text, voice, background apps, and available RAM. The browser tests used approximately 0.84-1.23 GB peak worker RSS; this excludes browser/interface memory and does not measure total committed memory. This PC does not generate speech in real time.
- The 5,000-character input limit and chunk preservation are covered by focused tests. The actual long-form browser run covers two passages, not every possible 5,000-character passage or pronunciation.
- Gradio 5.49.1 has a duplicate-upload race on Windows: its background move can truncate an identical cached file while the browser reads it. `studio/uploads.py` reuses the existing content-addressed upload instead. Both a real repeated-upload browser retest and focused file-read checks pass. The compatibility fix is scoped to that pinned Gradio version.
