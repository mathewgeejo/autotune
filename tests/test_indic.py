import numpy as np
import pytest

from studio.indic import IndicEngine


@pytest.mark.parametrize('rate,expected_total', [(1, 200), (2, 150), (0.5, 300)])
def test_speaking_rate_changes_new_speech_without_rescaling_the_reference(rate, expected_total):
    original = np.broadcast_to(np.arange(200, dtype=np.float32)[None, :, None], (1, 200, 100)).copy()
    class PublishedEncoder:
        calls = 0
        def run(self, _, feeds):
            self.calls += 1
            # A second graph call at a different speed would shift the prompt
            # embeddings even if the final tensor had the correct total length.
            assert float(feeds['speed']) == 1.0
            return [original.copy()]
    engine = IndicEngine.__new__(IndicEngine)
    engine.text_encoder = PublishedEncoder()
    engine.text_ids = lambda text: np.array([[1]], dtype=np.int64)
    engine.speed = rate
    condition = engine.text_conditioning('A new phrase.', np.array([[1]], dtype=np.int64), 100)
    assert condition.shape[1] == expected_total
    assert condition.shape[1]-100 == pytest.approx(100/rate)
    np.testing.assert_array_equal(condition[:, :100], original[:, :100])
    assert condition[0, 100, 0] == 100
    assert condition[0, -1, 0] == 199
    assert engine.text_encoder.calls == 1
