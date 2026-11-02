# Voice Studio completion gates

Cloud deployment is the final step. No paid plan, billing upgrade, or charge is authorized.

- [ ] Near-indistinguishable quality: independent speaker verification, content recognition, audio diagnostics, and a blind listening comparison; report measured failures, never relabel a similarity score as a percentage of identity.
- [ ] Real Malayalam, Hindi, and English cloning using supported models, native-script text, and reference-conditioned speech; test each language and cross-language behavior.
- [ ] One-minute uploads: exact 60-second boundary, silence/noise/stereo/compressed files, useful selection feedback, no claim that a model consumes all 60 seconds if its conditioning window is shorter.
- [ ] Built-in analysis: pitch and voicing, energy/loudness, clipping/noise indicators, spectral characteristics, speaker similarity, text errors, plots, time-localized differences, reproducible JSON/CSV export.
- [ ] Responsive UI: phone/tablet/desktop, keyboard/accessibility, microphone and upload, A/B playback, progress/cancel/recovery, analysis explanations and model/download status.
- [ ] Self-hosting: configurable data directory/host/port, Linux and Windows workers, Docker and Compose, persistent cache, documented HTTPS and private access.
- [ ] Tests: unit and adversarial inputs, regression, independent quality metrics, native-language samples, 60-second input and output, browser/device/accessibility checks, clean installation, Linux container runtime.
- [ ] Final deployment: verify an authenticated hosting account and a genuinely zero-cost plan with sufficient RAM, validate runtime first, deploy only after the above gates, verify the resulting public URL without a billing upgrade.

Current environment: Windows CPU, about 8 GB RAM; Docker and WSL are not installed. Do not claim a Docker runtime test until a real engine/container has been run.

## Evidence collected

- Implemented native-script engine routing, Indic automatic reference transcription, acoustic diagnostics, independent audit, exports, portable worker cancellation, configurable storage/auth/networking, Docker/Compose, and Linux/Windows CI definitions.
- 20 regression checks pass. Actual minute-long WAV/FLAC/MP3/M4A input checks and rejection of 61 seconds pass. Updated English browser upload and simulated microphone generation pass.
- Responsive analysis/export checks pass at four widths. Axe WCAG 2.1 A/AA checks pass on both tabs at desktop/phone widths after contrast, table semantics, and placeholder repairs.
- Genuine cloning quality failures remain recorded; the generic recognizer also failed on natural Malayalam. Script validation prevents misleading word-error reports, and a native Malayalam recognizer is being calibrated. Failed tone-conversion refinement was removed.
- Current Git author is `mathewgeejo <mathewgeejo2006@gmail.com>` globally and locally. The target GitHub sign-in is not verified; publishing must not use the old account.
- No deployment, billing upgrade, paid API call, or container-build claim has been made. Docker runtime and cloud access remain unresolved.
