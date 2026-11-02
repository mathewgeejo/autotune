# Voice Studio completion gates

Cloud deployment is the final step. No paid plan, billing upgrade, or charge is authorized.

- [ ] Near-indistinguishable quality: independent speaker verification, content recognition, audio diagnostics, and a blind listening comparison; report measured failures, never relabel a similarity score as a percentage of identity.
- [x] Functional Malayalam, Hindi, and English generation using supported models, native-script text, and reference conditioning. Native reference tests run; cross-language failures recorded. Perceptual quality remains the separate open gate above.
- [x] One-minute uploads: exact 60-second boundary and silence/noise diagnostics, stereo/compressed inputs, selection feedback showing the ten-second conditioning window.
- [x] Built-in analysis: pitch and voicing, energy/loudness, clipping/noise indicators, spectra, independent speaker similarity, text errors, plots, localized differences, JSON/CSV/ZIP export. Full audit verified through the browser with the real Malayalam output.
- [x] Responsive UI checks: desktop/tablet/phone viewports, keyboard/automated accessibility, simulated microphone and uploads, reference/output playback, progress/cancel/recovery, analysis explanations and download status. Physical-device and human screen-reader coverage remains limited.
- [ ] Self-hosting: configurable data directory/host/port, Linux and Windows workers, Docker and Compose, persistent cache, documented HTTPS and private access.
- [ ] Tests: unit and adversarial inputs, regression, independent quality metrics, native-language samples, 60-second input and output, browser/device/accessibility checks, clean installation, Linux container runtime.
- [ ] Final deployment: verify an authenticated hosting account and a genuinely zero-cost plan with sufficient RAM, validate runtime first, deploy only after the above gates, verify the resulting public URL without a billing upgrade.

Current environment: Windows CPU, about 8 GB RAM; Docker and WSL are not installed. Do not claim a Docker runtime test until a real engine/container has been run.

## Evidence collected

- Implemented native-script engine routing, Indic automatic reference transcription, acoustic diagnostics, independent audit, exports, portable worker cancellation, configurable storage/auth/networking, Docker/Compose, and Linux/Windows CI definitions.
- 26 regression checks pass. Actual minute-long WAV/FLAC/MP3/M4A input checks and rejection of 61 seconds pass. Updated English browser upload and simulated microphone generation pass. A 71.84-second six-passage output preserved every sample in order.
- Responsive analysis/export checks pass at four widths. Axe WCAG 2.1 A/AA checks pass on both tabs at desktop/phone widths after contrast, table semantics, and placeholder repairs.
- Mobile WebKit uploads/analysis/exports pass with no JavaScript errors. Instance authentication and download boundaries pass. Compose's native parser validates the configuration; it does not establish container runtime compatibility.
- Automatic offline Hindi and Malayalam workflows need no supplied reference words or reference-language selection. Both sampled outputs have zero recognized content errors; independent speaker cosine differs substantially (Hindi 0.951, Malayalam 0.415), so quality is not universally verified.
- Genuine cloning quality failures remain recorded; the generic recognizer also failed on natural Malayalam. Script validation prevents misleading word-error reports. Native Malayalam calibration and a fully automatic offline cloning run now pass script/content checks on the sampled phrase. Speaker similarity remains limited. Unsupported language detections are routed before word decoding. Failed tone-conversion refinement was removed.
- Current Git author is `mathewgeejo <mathewgeejo2006@gmail.com>` globally and locally. The target GitHub sign-in is not verified; publishing must not use the old account.
- No deployment, billing upgrade, paid API call, or container-build claim has been made. Docker runtime and cloud access remain unresolved.
