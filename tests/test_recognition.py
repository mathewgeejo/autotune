import numpy as np
import pytest

from studio.recognition import evaluation_level, transcript_script_check, content_comparison


def test_incompatible_asr_script_is_not_scored_as_a_synthesis_error():
    wrong = transcript_script_check('સહ ਗੁਸਤਿ ਕਾਰੁਮ ਲੂਣਾ', 'ml')
    assert not wrong['reliable']
    result = content_comparison('സഹ ഗുസ്തിക്കാരും', {'text': 'સહ ਗੁਸਤਿ', 'script_check': wrong})
    assert not result['available']
    assert 'word_error_rate' not in result
    assert transcript_script_check('നമസ്കാരം ഇന്ന് നല്ല ദിവസമാണ്', 'ml')['reliable']
    assert transcript_script_check('नमस्ते आज मौसम अच्छा है', 'hi')['reliable']
    assert not transcript_script_check('', 'hi')['reliable']
    assert transcript_script_check('आज API ठीक है', 'hi')['reliable']


def test_level_matching_does_not_change_the_waveform_or_clip():
    t = np.arange(16000)/16000
    original = 0.001*np.sin(2*np.pi*180*t).astype(np.float32)
    adjusted, metadata = evaluation_level(original)
    assert np.sqrt(np.mean(adjusted*adjusted)) == pytest.approx(0.08, abs=1e-6)
    nonzero = np.abs(original) > 1e-6
    assert np.std(adjusted[nonzero]/original[nonzero]) < 1e-4
    assert metadata['gain_db'] > 30
    transient = np.r_[np.zeros(10000), 1.0].astype(np.float32)
    normalized, _ = evaluation_level(transient)
    assert np.max(np.abs(normalized)) <= 0.950001
    silent, _ = evaluation_level(np.zeros(16000))
    assert not np.any(silent)


def test_auto_reference_recovers_wrong_script_before_caching(tmp_path, monkeypatch):
    from studio import worker, recognition
    monkeypatch.setattr(worker, 'REFERENCES', tmp_path/'cache')
    preview = tmp_path/'reference.wav'
    preview.write_bytes(b'test audio fingerprint')
    calls = []
    def recognize(path, folder, language, progress, fallback_language=None):
        calls.append(language)
        text, detected = ('സഹ ഗുസ്തിക്കാരും', 'ml') if language == 'ml' else ('સહ ગੁસ્તિ', 'gu')
        return {'text': text, 'language': detected, 'script_check': transcript_script_check(text, detected)}
    monkeypatch.setattr(recognition, 'recognize', recognize)
    result = worker.reference_transcript(preview, tmp_path, None, lambda message: None, 'ml')
    assert result == 'സഹ ഗുസ്തിക്കാരും'
    assert calls == [None, 'ml']
    assert worker.reference_transcript(preview, tmp_path, None, lambda message: None, 'ml') == result
    assert calls == [None, 'ml']


def test_malayalam_hint_routes_before_decoding_an_unsupported_language(tmp_path, monkeypatch):
    import soundfile as sf
    import faster_whisper
    from types import SimpleNamespace
    from studio import recognition
    source = tmp_path/'reference.wav'
    sf.write(source, np.sin(2*np.pi*180*np.arange(48000)/16000)*0.1, 16000)
    monkeypatch.setattr(recognition, 'decode', lambda *args, **kwargs: source)
    monkeypatch.setattr(recognition, 'download_asr', lambda progress, language: str(language))
    calls = []
    class Recognizer:
        def __init__(self, root, **kwargs):
            self.root = root
        def transcribe(self, audio, language=None, **kwargs):
            calls.append(language)
            if language is None:
                def expensive_wrong_language():
                    raise AssertionError('Do not decode the wrong language before routing to the native model')
                    yield
                return expensive_wrong_language(), SimpleNamespace(language='gu', language_probability=0.8)
            segment = SimpleNamespace(start=0, end=3, text='നമസ്കാരം', avg_logprob=-0.1, words=[])
            return iter([segment]), SimpleNamespace(language='ml', language_probability=1)
    monkeypatch.setattr(faster_whisper, 'WhisperModel', Recognizer)
    result = recognition.recognize(source, tmp_path, fallback_language='ml')
    assert calls == [None, 'ml']
    assert result['text'] == 'നമസ്കാരം'
    assert result['script_check']['reliable']
    assert result['initial_language_detection']['language'] == 'gu'
