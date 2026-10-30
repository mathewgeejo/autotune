# Verification — 29 October 2026

Tested locally on Windows, Intel Core i3-1115G4, approximately 8 GB RAM, CPU inference.

| Flow | Result | Evidence |
| --- | --- | --- |
| Focused tests | Passed | Six tests: trimming and limits; silence/corrupt input; reference selection; invalid trim numbers; cancellation of parent/child processes including interrupted JSON writes; storage deletion preserving models and rejecting outside paths. |
| Instrument engine | Passed | Actual Basic Pitch ONNX inference detected the four expected MIDI notes (57, 60, 64, 69). FluidSynth rendered a nonempty WAV and MIDI export. |
| Browser upload → output | Passed | Chrome uploaded a WAV, ran the actual engine, rendered the output waveform and WAV/MIDI links, opened Singer controls, and rendered mobile layout without JavaScript errors. Warm conversion: 14 seconds; sampled worker memory peak: 206 MB. |
| Browser microphone → output | Passed | Chrome used a simulated microphone, recorded audio, uploaded it, and completed actual instrument conversion without JavaScript errors. No physical microphone was accessed. |
| Vocal extraction | Passed | Demucs separated a 14.56-second reference and automatically selected a 10-second vocal section. 110 seconds elapsed; approximately 976 MB sampled worker memory peak. |
| Song link and cache reuse | Passed | Imported a public direct audio URL with yt-dlp, decoded it, and reused its cached vocal reference. Completed in 9.3 seconds. |
| Singer engine | Passed | Actual Seed-VC singing model, CPU, 5-second input/reference, Preview/10 steps. Output: 4.992-second WAV at 44,100 Hz, finite samples, RMS 0.344, peak 0.950. |
| Singer performance | Measured | Approximately 191.6 seconds inference, excluding loading/download waits. Total test elapsed 688 seconds because it also waited for the remaining model download. Sampled worker memory peak: approximately 1,600 MB RSS; this does not measure committed or swapped memory. |
| Offline model readiness | Passed | All required files resolved from cache with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. File-based transformations use these local engines; importing links still needs internet. |
| Setup and static checks | Passed | All 129 installed packages compatible; locked dependency installation dry-run requires no changes; PowerShell setup parses; asset setup reruns successfully; Ruff correctness checks pass. |

No project builds were run. The app runs directly with Python.

## Limits of these checks

- Singing tests used upstream example audio. Human assessment of similarity to a particular singer and quality on the user's own microphone recordings remains necessary.
- Link import was verified with a public direct WAV URL. YouTube and other streaming services can impose login/download restrictions or change their extractors; the app provides an upload fallback.
- Vocal separation used an upstream vocal example. Dense accompaniment, multiple singers, and instrumental-only material can affect automatic reference selection.
- Instrument inference was checked with a controlled melody and a simulated microphone. Actual imitations and noisy recordings will vary in note accuracy.
- Desktop and mobile screenshots are saved locally under `data/smoke/`.
