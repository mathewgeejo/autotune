# English and Malayalam quality focus

Current priority: improve actual voice similarity and naturalness in English and Malayalam, optimize runtime, and validate with real model outputs before infrastructure or deployment work. Do not mark perceptual indistinguishability established from cosine or word-error scores alone.

- Preserve all baseline and failed trials; compare changes against the same references and sentences.
- Expand natural Malayalam references and English voices; calibrate independent judges with human-audio controls.
- Investigate reference artifacts, conditioning, guidance/decoding, and stronger supported models.
- Report content accuracy, independent voice similarity, pitch/timing, latency, and memory separately.
- Keep builds out of the iteration loop. Use focused tests and real synthesis comparisons.
- Commit and push prepared changes about every ten minutes using `mathewgeejo <mathewgeejo2006@gmail.com>`; record any push failure without using another account.

First finding: the current Malayalam reference has a 0.462 peak only 9 ms into the file, followed by about 360 ms of low energy before speech. The isolated click dominates peak normalization. Boundary-transient selection is being tested; full synthesis quality remains unproven.

The first controlled comparison is measured in `QUALITY_RESULTS.md`: 0.879 → 0.921 independent speaker cosine against the same cleaned reference, with zero recognized content errors on the sampled phrase. This is a promising short-sample result, not a universal perceptual pass.

Git: target account now appears in Git Credential Manager, and noninteractive `git push --dry-run origin HEAD` succeeds. GitHub CLI has separate authentication state and is not used for pushes.

Current checkpoint: English waveform decoding defaults to up to four logical threads after two voices and a reverse-order comparison produced faster, identical WAVs. The normal longer worker test passes 54 words across two passages with zero recognized errors. General/English language-model inference stays at two threads while its promising four-thread stage comparison receives broader checks.

Follow-up: the second English voice has slower autoregression with four threads, so language-model default two stays. The 64-step Malayalam control has unchanged recognized wording. Original public IndicF5 at guidance 2/3 did not beat the existing higher-guidance DhVaani reading's wording result and is not integrated. Its unused weights/runtime were removed at the user's storage-cleanup request; retain the measured reports and audio.

Malayalam pronunciation remains open. The difficult sentence has the same default word-error rate on CPU and Intel. A guidance-3 retry improves its recognized words but can lose voice similarity, so the optional word-check path compares both readings and keeps the default if the relative voice guard fails. A cold GPU variation was reproduced and two fresh-process first-step warm-up trials match the earlier good recording exactly. The warm-up is integrated only into that optional retry; the actual warmed worker now selects the same recording, halving recognized word/grapheme errors within the unchanged voice guard. Broader voice/shape coverage and the browser no-retry case remain open. The checked run is slow under low available RAM; future recognition reports separate loading and decoding costs. Plain Detailed remains the UI default. Two same-words cases have calibrated content/acoustic checks and a prepared, unreviewed listening pack. Do not count its browser checks as human listening evidence.

English follow-up: paired Q8 reconstruction fixes the recognized name and reduces pitch/spectral differences on Dave, with almost identical speaker cosine, but costs 760.6 seconds versus Q4's 25.1. The first full OpenVINO CPU render takes 228.8 seconds with zero recognized errors but lower speaker cosine 0.9088, so it is not adopted. Accuracy execution reduces sampled logit error from about 1.5% to below 2e-6 relative norm; a full accuracy-mode render is running with a recorded ORT seed. Padding is rejected after real incremental tokens disagree, and removed from the rendering CLI. Verify decoder reproducibility and broader voice/content quality before changing production Q4.

Recognizer storage follow-up: full INT8 compaction regresses two of six real-audio controls and is rejected. Retaining original FP32 convolution weights yields six exact matches with a 0.73 GiB checkpoint, but it is not selected by the app yet. Next gates are a same-condition original-model load/memory control, integration validation and the browser no-retry case. The original source and all evidence remain available; automatic approval review blocked deleting the rejected candidate weights. Do not substitute these ASR checks for synthesis naturalness or human listening.
