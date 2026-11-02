# Additional engine and analysis components

- English synthesis: ResembleAI/Chatterbox Turbo ONNX, MIT. Revision and model provenance are pinned in `studio/models.py`.
- Malayalam/Hindi synthesis: DhVaani-0.5, originally released by ARTPARK-IISc (Project Vaani), built on k2-fsa/ZipVoice. The public ONNX export `Bbkblo/DhVaani-0.5-ONNX` declares Apache-2.0 and is pinned to `597cd20e44e85910fdb086d04c074cdc313a1a80`. Its documented DSP and inference contract are adapted in `studio/indic.py`; the downloaded export's README is retained with the weights. This is a third-party export, not an official ARTPARK ONNX release. Reference: https://huggingface.co/Bbkblo/DhVaani-0.5-ONNX and https://vaani.iisc.ac.in/models.
- Vocos waveform decoding: charactr/vocos-mel-24khz, MIT, included in the pinned DhVaani export.
- Independent speaker verification: Microsoft WavLM base-plus speaker verification, public ONNX conversion by Xenova. It is separate from the voice-cloning engine; its scores do not establish perceptual indistinguishability.
- Independent transcription: Whisper large-v3-turbo, public CTranslate2 conversion by mobiuslabsgmbh, pinned in `studio/models.py`. No expected-text prompt is supplied when measuring content errors.
- Native Malayalam transcription: `smcproject/vegam-whisper-medium-ml`, MIT CTranslate2 conversion of `thennal/whisper-medium-ml`, revision `21334192f79d87cc05fe4253d7d839b5a42c33bf`. It uses the pinned multilingual `openai/whisper-tiny` tokenizer, not the incompatible large-v3 vocabulary. Model card: https://huggingface.co/smcproject/vegam-whisper-medium-ml. Its training data includes FLEURS, so FLEURS clips are calibration examples and are not claimed to be independent held-out tests of this recognizer.
- Acoustic analysis: NumPy/SciPy, BS.1770 loudness via pyloudnorm, Plotly charts. Pitch and spectral comparisons are descriptive diagnostics, not clinical assessments or claims of speaker identity.

Upstream licenses and model cards apply. No paid inference API is used.

Upstream clarification checked 2 November 2026: although the DhVaani ONNX export declares Apache-2.0, the original maintainer states that the pretrained ZipVoice weights carry CC-BY-NC restrictions inherited from Emilia. Treat the voice weights as non-commercial; the code license does not override weight terms. Source: https://huggingface.co/ARTPARK-IISc/DhVaani-0.5/discussions/1.

Optional quality experiments use Microsoft's DNSMOS P.835 model, CC-BY-4.0, from `microsoft/DNS-Challenge` revision `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`, with the upstream calibration and short-clip repetition protocol. Model SHA256 is pinned in `scripts/quality_proxies.py`, and the license is retained with the cached model. It is a noise-quality proxy, not a human listening result or an identity/naturalness guarantee. Source: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS.
