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

Malayalam pronunciation remains open. The difficult sentence has the same word-error rate on CPU and Intel; the public F5 candidate also fails to improve it and is not integrated. The reference-fixed rate candidate was rejected and moved out of production. Next measured candidates include a 64-step flow control, additional guidance/conditioning comparisons if justified, and broader novel text. Two same-words cases now have calibrated content/acoustic checks and a prepared, unreviewed listening pack. Do not count its browser checks as human listening evidence.
