# Additional engine and analysis components

- English synthesis: ResembleAI/Chatterbox Turbo ONNX, MIT. Revision and model provenance are pinned in `studio/models.py`.
- Malayalam/Hindi synthesis: DhVaani-0.5, originally released by ARTPARK-IISc (Project Vaani), built on k2-fsa/ZipVoice. The public ONNX export `Bbkblo/DhVaani-0.5-ONNX` declares Apache-2.0 and is pinned to `597cd20e44e85910fdb086d04c074cdc313a1a80`. Its documented DSP and inference contract are adapted in `studio/indic.py`; the downloaded export's README is retained with the weights. This is a third-party export, not an official ARTPARK ONNX release. Reference: https://huggingface.co/Bbkblo/DhVaani-0.5-ONNX and https://vaani.iisc.ac.in/models.
- Vocos waveform decoding: charactr/vocos-mel-24khz, MIT, included in the pinned DhVaani export.
- Independent speaker verification: Microsoft WavLM base-plus speaker verification, public ONNX conversion by Xenova. It is separate from the voice-cloning engine; its scores do not establish perceptual indistinguishability.
- Independent transcription: Whisper large-v3-turbo, public CTranslate2 conversion by mobiuslabsgmbh, pinned in `studio/models.py`. No expected-text prompt is supplied when measuring content errors.
- Acoustic analysis: NumPy/SciPy, BS.1770 loudness via pyloudnorm, Plotly charts. Pitch and spectral comparisons are descriptive diagnostics, not clinical assessments or claims of speaker identity.

Upstream licenses and model cards apply. No paid inference API is used.
