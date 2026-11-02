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
