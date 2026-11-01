import numpy as np
import pytest
import soundfile as sf

from studio.audio import AudioError, best_reference, decode, validate_signal


def tone(path, seconds, rate=16000):
    t = np.arange(int(seconds*rate))/rate
    sf.write(path, 0.2*np.sin(2*np.pi*220*t), rate)
    return path


def test_decoder_preserves_selected_duration(tmp_path):
    source = tone(tmp_path/'long.wav', 32)
    result = decode(source, tmp_path/'clip.wav', start=4, duration=5, rate=24000)
    assert sf.info(result).duration == pytest.approx(5, abs=0.02)
    assert sf.info(result).samplerate == 24000


def test_silence_and_corrupt_audio_have_useful_errors(tmp_path):
    silent = tmp_path/'silent.wav'
    sf.write(silent, np.zeros(16000), 16000)
    with pytest.raises(AudioError, match='silent'):
        validate_signal(silent)
    bad = tmp_path/'broken.mp3'
    bad.write_bytes(b'not audio')
    with pytest.raises(AudioError, match='Could not read'):
        decode(bad, tmp_path/'bad.wav')


def test_reference_picker_avoids_silent_intro(tmp_path):
    path = tmp_path/'reference.wav'
    rate = 16000
    t = np.arange(12*rate)/rate
    sf.write(path, np.r_[np.zeros(8*rate), 0.25*np.sin(2*np.pi*220*t)], rate)
    start, length, total = best_reference(path, seconds=5)
    assert start >= 8
    assert length == 5
    assert total == 20
