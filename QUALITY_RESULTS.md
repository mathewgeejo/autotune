# English and Malayalam voice-quality work

This is the current priority. Deployment and Docker work are deferred. These measurements do not establish human indistinguishability.

## 2 November: reference transient and fair Malayalam comparison

The original Malayalam FLEURS recording contains an isolated peak at 9 ms, separated from speech by a long quiet interval. It constrained reference normalization and the independent verifier's input level. The new selector excludes isolated, extreme boundary transients while preserving speech and interior pauses. It does not denoise or pitch-shift the voice.

Fixed model and sentence: DhVaani FP32, 32 steps, guidance 2, seed 666; “നമസ്കാരം. ഇന്ന് നല്ല ദിവസമാണ്.”

| Comparison | Independent WavLM cosine |
| --- | ---: |
| Previous output against the same cleaned natural reference | 0.8790 |
| New output against that cleaned reference | 0.9214 |
| Two halves of the cleaned human recording | 0.9196 |

The candidate has zero recognized word/grapheme errors on this short phrase. Request time was 141.6 seconds; peak worker RSS was 881 MB. Selected natural speech: 5.2 seconds, starting at 0.26 seconds. Baseline output is retained, not replaced.

The apparent change from older scores around 0.45 to 0.92 must **not** be reported as a pure synthesis improvement: changing the evaluator's reference explains much of it. The fair same-reference synthesis comparison is **0.879 → 0.921**. The current verifier also excludes the boundary click before matching listening level: against the old raw reference it measures baseline 0.8835 and candidate 0.9229.

Evidence: `data/quality/ml-edge-trial.json`, `data/quality/verifier-v2-control/report.json`, and job `2f0c6736dac4434c957df0ecd9cd60d0`. These are local audio/measurement artifacts excluded from Git. Checkpoint source and tests are versioned.

Next: independent English baseline; multiple Malayalam references and longer sentences; controlled step-count/decoding optimization; comparisons with stronger supported models when they address a measured failure. Preserve difficult and failed examples, calibrate every judge, and prepare listening comparisons before claiming the requested perceptual result.

## English baseline and initial speed sweep

The new Dave English run has zero recognized content errors and independent speaker cosine 0.693; same-human/different-word control is 0.713. Request time was 25.7 seconds, peak worker RSS 881 MB. The expressive reference makes this a difficult identity control; retain it alongside steadier references. Evidence: `data/quality/en-dave-baseline.json`.

On the fixed cleaned Malayalam reference, FP32/guidance-2 trials measured:

| Flow steps | Synthesis seconds | Speaker cosine | Recognized word errors |
| --- | ---: | ---: | ---: |
| 8 | 36.3 | 0.9368 | 0 |
| 16 | 67.1 | 0.9192 | 0 |
| 32 baseline | about 133 | 0.9214 | 0 |

These are candidate optimizations. Defaults remain unchanged until more voices, longer text, and acoustic/naturalness checks support the change. Evidence: `data/quality/ml-steps-sweep.json`.

Corpus coverage: all 958 rows in the pinned Malayalam FLEURS test metadata are labeled FEMALE. The collector records missing labels instead of falsely claiming balanced coverage. Gender labels are not speaker IDs. Revision: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.

## Quality guards and a failure found by broader testing

The optional DNSMOS P.835 noise-quality proxy was calibrated on the natural reference. Overall predictions were 3.15 for the natural recording, 3.21 for the 32-step candidate, and about 3.05 for the 8/16-step candidates. These are not human MOS ratings or naturalness/identity proof. Short inputs are repeated using the upstream protocol, and repetition is explicitly recorded. Keep 32-step defaults while broader comparisons run; cosine and word correctness alone do not rule out audio-quality regressions. Evidence: `data/quality/ml-step-noise-quality.json`.

The second Malayalam reference hit a Windows `PermissionError` replacing `status.json` after 30 diffusion steps. It was a progress-file race, not a measured synthesis-quality failure. Atomic JSON writes now retry transient locks, and an advisory status update cannot discard the audio request. Result/failure files remain strict. Both focused regression checks pass; the failed job `34fc1e04f2494d238968089554f14190` is retained for diagnosis.

Stronger-model investigation: the public IndicF5 ONNX DiT export at `developerabu/IndicF5-ONNX@8886363d9473330618a458b119d5d8bd95ffccba` has fixed `[1,128,100]` audio/conditioning and `[1,128]` text inputs. That cannot directly represent normal 3–10-second references. It has not been substituted into the working engine or counted as a successful clone.

## Broader reference coverage

The retry on the second FLEURS recording completes after the status-lock fix: zero recognized content errors, independent speaker cosine **0.9193**, 212.3-second request, and about 1,048 MB peak worker RSS. Its same-recording half-to-half control is only 0.4257, another reason to avoid a universal cosine cutoff. Evidence: `data/quality/ml-second-reference-retry.json`.

Two additional Malayalam recordings from OpenSLR SLR63's male archive are now collected, with different anonymized filename groups. They were selected by duration and group, not synthesis scores. HTTP byte ranges fetched 4.31 MB instead of the full 635.66 MB ZIP; per-file hashes and CC-BY-SA attribution are retained in `data/quality/corpus/openslr63-male/manifest.json`. Filename groups are an inference about recording structure, not independent proof of speaker identity.

The regression suite now passes **31 tests**. Checkpoints are pushed to `mathewgeejo/autotune`; no application builds or deployment work are part of this quality iteration.

Per-passage stage timings are now recorded in `speech-metrics.json`: English language-model versus waveform-decoder time; Indic text-conditioning, flow, and waveform-decoder time. This adds measurement without changing sampling or inference precision. Seven relevant speech/Indic checks and Ruff correctness checks pass. Additional male-archive and Jo English comparisons are underway.
