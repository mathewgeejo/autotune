import json

import numpy as np
import pytest

from studio.analysis import acoustic_profile, compare_profiles, transcript_errors


def voice_tone(frequency=180, seconds=2, amplitude=0.2, rate=16000):
    time = np.arange(int(seconds*rate))/rate
    return amplitude*(np.sin(2*np.pi*frequency*time)+0.2*np.sin(4*np.pi*frequency*time))


def test_pitch_and_gain_diagnostics_detect_known_changes():
    reference = acoustic_profile(voice_tone(180))
    output = acoustic_profile(voice_tone(240, amplitude=0.1))
    assert reference['metrics']['median_pitch_hz'] == pytest.approx(180, abs=1)
    assert output['metrics']['median_pitch_hz'] == pytest.approx(240, abs=1)
    result = compare_profiles(reference, output)
    assert result['differences']['median_pitch_hz'] == pytest.approx(60, abs=1)
    assert output['metrics']['rms_dbfs']-reference['metrics']['rms_dbfs'] == pytest.approx(-6.02, abs=0.02)
    assert not result['aligned_comparison']['available']
    json.dumps(result, allow_nan=False)


def test_identical_words_audio_has_zero_alignment_error():
    profile = acoustic_profile(voice_tone())
    result = compare_profiles(profile, profile, same_words=True)['aligned_comparison']
    assert result['available']
    assert result['mean_spectral_distance'] == pytest.approx(0)
    assert result['median_absolute_pitch_difference_cents'] == pytest.approx(0)


def test_silence_noise_and_clipping_are_not_good_voice_matches():
    silent = acoustic_profile(np.zeros(16000))
    assert silent['metrics']['median_pitch_hz'] is None
    assert silent['metrics']['silence_percent'] == 100
    assert silent['warnings']
    noise = acoustic_profile(np.random.default_rng(8).normal(0, 0.05, 32000))
    assert noise['metrics']['voiced_percent'] < 5
    clipped = acoustic_profile(np.clip(voice_tone(amplitude=2), -1, 1))
    assert clipped['metrics']['clipped_percent'] > 20
    assert any('Clipping' in warning for warning in clipped['warnings'])
    json.dumps(compare_profiles(silent, noise), allow_nan=False)


def test_multilingual_content_errors_preserve_combining_characters():
    hindi = transcript_errors('नमस्ते! आज मौसम अच्छा है।', 'नमस्ते आज मौसम अच्छा है')
    assert hindi['word_error_rate'] == 0
    assert hindi['grapheme_error_rate'] == 0
    malayalam = transcript_errors('നമസ്കാരം. ഇന്ന് നല്ല ദിവസമാണ്.', 'നമസ്കാരം ഇന്ന് നല്ല ദിവസമാണ്')
    assert malayalam['word_error_rate'] == 0
    wrong = transcript_errors('The blue notebook is here.', 'The red notebook is here.')
    assert wrong['word_error_rate'] == pytest.approx(0.2)
    assert wrong['differences'][0]['expected'] == 'blue'
    assert wrong['differences'][0]['recognized'] == 'red'


def test_full_minute_is_accepted_and_over_limit_rejected():
    from studio.audio import AudioError
    minute = acoustic_profile(voice_tone(seconds=60))
    assert minute['metrics']['duration_seconds'] == 60
    with pytest.raises(AudioError, match='60 seconds'):
        acoustic_profile(voice_tone(seconds=60.1))
