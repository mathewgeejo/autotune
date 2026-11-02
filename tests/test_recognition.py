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
