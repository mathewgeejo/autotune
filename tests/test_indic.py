import numpy as np
import pytest

from studio.indic import IndicEngine


@pytest.mark.parametrize('rate,expected_total', [(1, 200), (2, 150), (0.5, 300)])
def test_speaking_rate_changes_new_speech_without_rescaling_the_reference(rate, expected_total):
    class PublishedEncoder:
        def run(self, _, feeds):
            # The upstream ONNX speed scales the full prompt+target length.
            return [np.zeros((1, round(200/float(feeds['speed'])), 100), dtype=np.float32)]
    engine = IndicEngine.__new__(IndicEngine)
    engine.text_encoder = PublishedEncoder()
    engine.text_ids = lambda text: np.array([[1]], dtype=np.int64)
    engine.speed = rate
    condition = engine.text_conditioning('A new phrase.', np.array([[1]], dtype=np.int64), 100)
    assert condition.shape[1] == expected_total
    assert condition.shape[1]-100 == pytest.approx(100/rate)
