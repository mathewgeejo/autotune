import numpy as np
import pytest

from studio.indic import IndicEngine


@pytest.mark.parametrize('rate,expected_total', [(1, 200), (2, 150), (0.5, 300)])
def test_speaking_rate_changes_target_duration_without_lengthening_reference_audio(rate, expected_total):
    original = np.broadcast_to(np.arange(200, dtype=np.float32)[None, :, None], (1, 200, 100)).copy()
    class PublishedEncoder:
        calls = 0
        def run(self, _, feeds):
            self.calls += 1
            # Exported speed expands the full text-conditioning sequence; the
            # real audio reference length stays fixed in the speech condition.
            indices = np.linspace(0, 199, round(200/float(feeds['speed']))).round().astype(np.int64)
            return [original[:, indices].copy()]
    engine = IndicEngine.__new__(IndicEngine)
    engine.text_encoder = PublishedEncoder()
    engine.text_ids = lambda text: np.array([[1]], dtype=np.int64)
    engine.speed = rate
    condition = engine.text_conditioning('A new phrase.', np.array([[1]], dtype=np.int64), 100)
    assert condition.shape[1] == expected_total
    assert condition.shape[1]-100 == pytest.approx(100/rate)
    assert condition[0, 0, 0] == 0
    assert condition[0, -1, 0] == 199
    assert engine.text_encoder.calls == (1 if rate == 1 else 2)
