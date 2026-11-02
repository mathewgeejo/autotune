import numpy as np
import pytest

from scripts import quality_proxies


@pytest.mark.parametrize('seconds', [2.5, 12])
def test_speech_quality_proxy_never_pads_or_batches_ragged_audio(monkeypatch, seconds):
    audio = np.ones(int(seconds*16000), dtype=np.float32)*0.1
    monkeypatch.setattr(quality_proxies, 'proxy_audio', lambda *args: ('fingerprint', audio, {}))
    inputs = []
    class Model:
        def run(self, outputs, feeds):
            wave = feeds['wave']
            assert wave.shape[0] == 1
            assert wave.dtype == np.float32
            assert np.all(wave == 0.1)
            inputs.append(wave.shape[1])
            return [np.array([3.5], dtype=np.float32)]
    proxy = quality_proxies.SpeechQualityProxy.__new__(quality_proxies.SpeechQualityProxy)
    proxy.session = Model()
    result = proxy.score('fixture.wav', 'unused')
    assert result['predicted_mos'] == 3.5
    assert inputs == ([40000] if seconds == 2.5 else [128000, 128000])


def test_speech_quality_proxy_refuses_insufficient_context(monkeypatch):
    monkeypatch.setattr(quality_proxies, 'proxy_audio', lambda *args: ('fingerprint', np.zeros(8000), {}))
    proxy = quality_proxies.SpeechQualityProxy.__new__(quality_proxies.SpeechQualityProxy)
    with pytest.raises(ValueError, match='one second'):
        proxy.score('fixture.wav', 'unused')
