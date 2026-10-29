import numpy as np
import pytest
import soundfile as sf

from studio.audio import AudioError, best_reference, trim_source, validate_signal


def tone(path, seconds, rate=16000):
    t = np.arange(int(seconds*rate))/rate
    sf.write(path, 0.2*np.sin(2*np.pi*220*t), rate)
    return path


def test_trim_preserves_selected_duration_and_rejects_long_recordings(tmp_path):
    source = tone(tmp_path/'long.wav', 32)
    with pytest.raises(AudioError, match='exceeds 30'):
        trim_source(source, tmp_path/'bad.wav')
    result = trim_source(source, tmp_path/'clip.wav', 4, 9)
    assert sf.info(result).duration == pytest.approx(5, abs=0.02)
    with pytest.raises(AudioError, match='end time'):
        trim_source(source, tmp_path/'bad.wav', 8, 3)


def test_silence_and_corrupt_audio_have_useful_errors(tmp_path):
    silent = tmp_path/'silent.wav'
    sf.write(silent, np.zeros(16000), 16000)
    with pytest.raises(AudioError, match='silent'):
        validate_signal(silent)
    bad = tmp_path/'broken.mp3'
    bad.write_bytes(b'not audio')
    with pytest.raises(AudioError, match='Could not read'):
        trim_source(bad, tmp_path/'bad.wav')


def test_reference_picker_avoids_silent_intro(tmp_path):
    path = tmp_path/'reference.wav'
    rate = 16000
    t = np.arange(12*rate)/rate
    sf.write(path, np.r_[np.zeros(8*rate), 0.25*np.sin(2*np.pi*220*t)], rate)
    start, length, total = best_reference(path, seconds=5)
    assert start >= 8
    assert length == 5
    assert total == 20


def test_trim_rejects_nan(tmp_path):
    with pytest.raises(AudioError, match='positive'):
        trim_source('unused', tmp_path/'out.wav', float('nan'), 0)
