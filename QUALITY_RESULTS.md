# English and Malayalam voice-quality work

This is the current priority. Deployment and Docker work are deferred. These measurements do not establish human indistinguishability.

Current candidate integration: optional Intel GPU acceleration for Detailed Indic flow, with automatic CPU fallback and the original FP32/32-step settings. The optional runtime is installed on this workstation. Normal-worker and longer-passage validation is in progress; English remains on its existing engine.

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

## Additional completed voices and measured bottlenecks

- OpenSLR filename group `00269`: speaker cosine **0.9298**, same-recording control **0.9325**, zero recognized content errors. Request 110.5 seconds, 766 MB peak worker RSS. Diffusion used 102.1 seconds; text conditioning and waveform decoding together used under 0.2 seconds. Evidence: `data/quality/ml-male-one-valid.json`.
- Jo English: speaker cosine **0.9143**, same-recording control **0.9631**, zero recognized content errors. Request 31.7 seconds, 892 MB peak worker RSS. Language generation used 6.9 seconds and waveform decoding 10.4 seconds. Evidence: `data/quality/en-jo-baseline.json`.

An optional Intel GPU experiment is now testing the **same FP32 flow model and 32 steps** against CPU output. It records waveform error/correlation, cold compilation, repeated synthesis, and independent content/identity checks. The package is installed in an isolated `.tools/openvino-runtime` folder; production dependencies and backend selection remain unchanged. No speed or fidelity result is claimed until the full trial finishes.

## Intel and synthesized-speech quality results

The full Intel FP32 trial completed: 23.8 seconds to compile, 96.3 seconds for the first synthesis and **82.6 seconds for the repeated synthesis**, versus about 133 seconds on CPU. Speaker cosine is **0.9243**, with zero recognized word/grapheme errors. CPU/GPU waveform correlations were 0.9934/0.9916; relative waveform RMS differences were 0.116/0.129, so this is not sample-identical arithmetic. Evidence: `data/quality/ml-intel-fp32.json`.

Independent quality guards remain close for this case: GPU DNSMOS overall 3.170 versus CPU 3.207; GPU UTMOS predicted MOS 3.084 versus CPU 3.071. Both are learned proxies, not human ratings. UTMOS correctly penalizes controlled noise and hard clipping on the English control (natural 3.773, noise 1.413, clipping 1.253). The natural Malayalam reference scores only 2.329, while the generated clip scores 3.071: source quality and language-domain effects make universal predicted-MOS cutoffs inappropriate. Jo's generated English scores 4.175. Evidence: `data/quality/utmos-calibration.json` and `data/quality/ml-intel-noise-quality.json`.

Production remains on CPU pending another voice and longer-passage acceleration checks. The experimental runtime stays optional and preserves the existing 32-step/full-precision quality settings.

The additional male-archive Intel trial completed: warm synthesis **50.7 seconds** versus CPU 102.3 seconds; speaker cosine 0.9044 versus CPU 0.9298, zero recognized content errors, CPU/GPU waveform correlation 0.9998. Cached compilation took 2.4 seconds. The optional backend now has CPU-only/unavailable-device checks, owned-output checks, and a failed-step CPU retry check. The complete regression suite passes **37 tests**. Normal-worker tests are the next integration gate. Evidence: `data/quality/ml-male-intel-fp32.json`.

## Production integration and longer-passage pronunciation

The normal production worker successfully used the optional Intel FP32 backend for a novel Malayalam sentence, retaining 32 steps and guidance 2. The 4.63-second result required 95.2 seconds for the request, with 847 MB peak worker RSS. Flow computation used 87.1 seconds. Backend integration is working; this does not close the voice-quality gate.

The longer sentence has independent speaker cosine **0.9749**, but recognized word error rate **0.50** and grapheme error rate **0.091**. A natural recording of the same expected words has word error rate **0.125** and grapheme error rate **0.023**. The natural recording is a different voice and is used only to calibrate content recognition. Three additional word differences in the synthesized recording remain a pronunciation-review target; high speaker similarity does not establish a pass. Evidence: `data/quality/ml-longer-production-intel.json`.

The trial runner now accepts a natural same-text content control and avoids asserting identical words when a longer reference has been cropped to ten seconds. CPU recovery is checked for both GPU exceptions and nonfinite output. The complete suite passes **38 tests**. Next: controlled guidance/timing comparisons on this difficult sentence, broader English passage testing, and fair listening comparisons. Human indistinguishability remains unproven.

## Speaking-rate conditioning correction

A real text-encoder probe found that changing the graph's speed moved reference text conditioning relative to the fixed reference audio. At speed 0.85, the old method changed the prompt region with relative tensor difference 1.341. Duration control now keeps the original prompt conditioning and resamples only the target region using existing embedding frames. In the same probe, prompt frames are exactly equal (relative difference zero), while total frames change from 892 to 969 with the 458-frame reference retained. Evidence: `data/quality/ml-rate-prefix-control.json`. This is a conditioning-contract check, not evidence that the pronunciation problem is solved.

The controlled sweep now pins its requested accelerator (CPU by default), records actual runtime settings, and compares guidance and target speaking rate as well as step count. Four full-precision/32-step trials on the difficult sentence are in progress. Completed synthesis results also retain a snapshot of their precision settings when the persistent engine is reused; later requests cannot rewrite earlier results in memory. Eight focused speech/Indic tests and Ruff correctness checks pass after these changes.

The four trials have now completed. None improves the difficult sentence's recognized word error rate (all 0.50). Guidance 2 / rate 1 remains best on grapheme error (0.091) and speaker cosine (0.9709). Guidance 2 / rate 0.85 measures 0.182 / 0.5728; guidance 1 / rate 1 measures 0.136 / 0.3025; guidance 1 / rate 0.85 measures 0.227 / 0.9590. The reference-preserving rate experiment is not a demonstrated voice-quality improvement, and quality defaults remain unchanged. The suite passes **39 tests**. Evidence: `data/quality/ml-long-pronunciation-sweep.json`. Its result metadata was restored from each run's original `speech-metrics.json` after finding the in-memory aliasing bug; the original combined report is retained separately.

An English decoder benchmark now captures the exact speech-token/speaker-feature inputs, measures repeated CPU runs, and attempts the unchanged quantized decoder on Intel. The first attempt stopped before GPU inference because OpenVINO rejects an external-weight symlink outside the graph's base directory. This is a model-file packaging issue; no English GPU speed/fidelity result is claimed. Evidence: `data/quality/en-jo-intel-decoder.json` and its log. Production English stays on CPU.

The model bundle fixes external-file loading without altering weights, but the GPU candidate fails during dynamic inference with an attention shape inconsistency. Specializing to the captured shapes instead fails GPU compilation on a transposed-convolution rank mismatch. Both failures are retained (`en-jo-intel-decoder-bundled.json`, `en-jo-intel-decoder-static.json`); neither is installed in production. A CPU OpenVINO candidate is being measured using the saved inputs, avoiding another language-model generation.

Repeated original-CPU decoding also reveals substantial waveform differences with unchanged inputs (correlation about 0.11–0.34). The exported graph contains four random-generation nodes, including vocoder noise/phase. Consequently, these uncontrolled waveform correlations cannot be interpreted as backend numerical error or identity regressions. Content/voice checks and any future comparison with explicitly captured random inputs must account for that stochastic control.

## Rejected rate implementation and stronger-model preparation

A direct comparison at rate 0.85 found the previous graph-duration method better than the reference-fixed target-frame candidate: word error **0.375 vs 0.50**, grapheme error **0.114 vs 0.182**, and speaker cosine **0.9108 vs 0.5728**. The graph-duration method is restored in production; normal rate remains 1. The rejected method stays available only as an explicit experimental sweep option. Tensor prefix equality was not sufficient evidence of better speech. Evidence: `data/quality/ml-long-legacy-rate.json`. Eight focused speech/Indic checks pass after restoration.

The OpenVINO CPU English candidate also fails compilation (`en-jo-openvino-cpu-static.json`). Production English remains on its working ONNX Runtime backend. Next optimization: measure CPU threading on the captured inputs instead of installing any failed backend.

An isolated stronger-model trial is being prepared for the publicly available `dheeyantra/dhee-indic-f5@b169ad1af92d659f9b85ac393510f8960e4a0690`. Tensor-header inspection confirms a 2546-row text embedding compatible in size with the cached 2545-character IndicF5 vocabulary; semantic compatibility still needs a real strict-load and synthesis test. The model has not been integrated or counted as a quality improvement. The bounded downloader records exact byte ranges and verifies the published full-file SHA256 before accepting the checkpoint.

English original-decoder threading: warm means of **9.110 seconds (2 threads), 8.179 seconds (4), and 10.778 seconds (1)** on Jo's captured inputs. All three last-repeat PCM WAVs have the same SHA256 (`3d4b7d2be8f0688865f4c8f55772bd5c54d6b4e92276be48d6a6bc1717e36b6a`), zero recognized word errors, and speaker cosine 0.8867. Random states vary between repeats but match across fresh sessions at corresponding repeat indices. A reverse-order timing check and the second English reference remain required before selecting a new default. Evidence: `data/quality/en-jo-decoder-threads.json`.

The reverse Jo check also favors four threads: **7.134 seconds vs 9.205 seconds** with two. A Dave comparison is running before a default change. Evidence: `data/quality/en-jo-decoder-threads-reverse.json`.

The full 1,402,784,904-byte F5 checkpoint is now verified against SHA256 `23d5cd75f5b52b54bd04efe39f871900146fa7123ae2c42fe5a38d04c7025738`. Preflight construction matches **all 364 transformer state keys and shapes**. The isolated Torch runtime sees 100 distributions including read-only app dependencies, with zero unsatisfied requirements in the runtime-visible metadata check. Vocos inference defers an unused BigVGAN/librosa import; no neural arithmetic is changed by that import adapter. Actual strict weight loading, synthesis and independent voice/content checks remain the next gates. Evidence: `data/quality/f5-runtime-preflight.json` and the model's `download.json`.

## English decoding optimization and first F5 audio result

Dave confirms the original-decoder thread result: **6.181 seconds with four threads vs 7.666 with two**. The last-repeat WAV is identical across both configurations (SHA256 `8beaf9d243690bb620f87d150487ec358d40bdf47087f3ffc83478f9148f8350`), with zero recognized word errors and speaker cosine 0.6987. On this laptop, normal English decoding now defaults to four logical threads; general sessions retain two. `VOICE_STUDIO_DECODER_THREADS` independently overrides decoding, and actual thread count is recorded per passage. A longer normal-worker passage test is in progress. Evidence: `data/quality/en-dave-decoder-threads.json`.

The isolated F5 trial completes strict transformer/vocoder weight loading and actual synthesis. FP32/32-step flow takes **334.4 seconds**, waveform decoding 0.206 seconds, with **1,497 MB peak worker RSS**. On the same difficult sentence/reference, speaker cosine is **0.8367**, word error **0.50**, and grapheme error **0.114**, compared with the current DhVaani candidate's 0.9749 / 0.50 / 0.091. It is not a demonstrated improvement and is not integrated. The natural same-words content control remains 0.125 word error. Evidence: `data/quality/ml-longer-f5-fp32.json`. These automated checks do not replace listening verification.

## Longer English production result and Malayalam backend control

The normal English worker completes **54 words / two passages / 17.04 seconds of audio**, with zero recognized word/grapheme errors and speaker cosine **0.9571**, compared with the same-recording human-half control 0.9631. The request takes 64.2 seconds, with 903.8 MB peak worker RSS. Both passage metrics confirm four decoder threads. Total synthesis is 54.5 seconds; language generation accounts for 30.8 seconds and waveform decoding 23.7 seconds. Evidence: `data/quality/en-jo-longer-production-threads4.json`. The full regression suite passes **39 tests** after the threading change.

The difficult Malayalam sentence now has a controlled original-CPU result: synthesis **146.4 seconds**, speaker cosine **0.9775**, word error **0.50**, and grapheme error **0.114**. Intel synthesis was 87.5 seconds with the same word error, grapheme error 0.091, and speaker cosine 0.9749. Waveform correlation is **0.99974**, relative RMS difference 0.02266. The word differences also occur on the original CPU backend; this test does not implicate Intel acceleration as their cause. Evidence: `data/quality/ml-longer-cpu-control.json`, `data/quality/ml-longer-backend-comparison.json`.

English autoregression threading is being profiled separately, requiring exact equality of the real decoder's speech-token, speaker-embedding and speaker-feature inputs. Its decoder is an explicit timing stub and does not produce or claim experimental speech audio. Same-words reconstruction/listening comparisons are being added alongside the already measured novel-text cases.

## Same-words controls and listening artifacts

Dave reconstruction: zero word/grapheme errors in both natural and generated recordings; speaker cosine **0.9201**. Natural/generated duration is 7.447/7.520 seconds. Median pitch is 154.4/137.7 Hz; median absolute aligned pitch difference is 149.6 cents. The higher same-text speaker score relative to novel-text Dave demonstrates content sensitivity, not a sudden universal identity improvement. Evidence: `data/quality/en-dave-same-words.json`.

Malayalam group 00269 reconstruction: speaker cosine **0.9694**, duration 4.880/4.885 seconds, median pitch 124.5/130.4 Hz, aligned pitch difference 132.0 cents. Generated recognized word error is 0.30; natural original is 0.20, and the exact selected conditioning preview is 0.30 with different word substitutions. Equal error totals do not establish identical pronunciation. The selected preview still recognizes the initial syllable that the generated reconstruction omits, so boundary trimming is not established as that omission's cause. Evidence: `data/quality/ml-male-one-same-words.json`, `data/quality/ml-selected-reference-content-control/report.json`.

A blind, level-matched listening pack for both same-text cases is prepared at `data/quality/listening-pairs-20261102/index.html`. Four audio clips load, all eight rating fields are present, and the 390-pixel viewport has no horizontal overflow or page errors. Source labels are kept in a separate answer key. **No human listening ratings have been entered**; preparation/browser checks are not perceptual results.

Initial Jo autoregression timing is 6.144 seconds (2 threads), 4.717 (4), and 6.736 (1), with exact equality of every real-decoder input across all runs. Production autoregression remains on two threads pending reverse-order and broader input checks. Evidence: `data/quality/en-jo-language-threads.json`. A 64-step FP32 control on the difficult Malayalam sentence is next; 32-step quality defaults remain unchanged.
