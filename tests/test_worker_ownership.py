import json
import weakref

import numpy as np

from studio import worker, speech, pronunciation


def test_cached_engine_is_released_before_word_check_even_in_serving_state(tmp_path, monkeypatch):
    class Engine:
        engine_key = ('indic', 'Detailed')
        def synthesize(self, *args, **kwargs):
            return {'audio': 'completed.wav'}
    state = {'engine': Engine()}
    previous = weakref.ref(state['engine'])
    monkeypatch.setattr(speech, 'prepare_sample', lambda *args, **kwargs: (np.zeros((1, 1)), tmp_path/'reference.wav', 'digest'))
    def checked(request, folder, prepared, result, progress, write):
        assert previous() is None, 'The serving frame must not retain its cached model during ASR'
        return result
    monkeypatch.setattr(pronunciation, 'check_pronunciation', checked)
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'mode': 'clone', 'source': 'source.wav', 'text': 'hello', 'language': 'ml',
                                  'reference_text': 'known words', 'quality': pronunciation.CHECKED_QUALITY}), encoding='utf-8')
    assert worker.run(request, state=state) is None
    assert not state
    assert (tmp_path/'result.json').exists()


def test_serving_state_retains_model_for_plain_warm_generation(tmp_path, monkeypatch):
    class Engine:
        engine_key = ('en',)
        calls = 0
        def synthesize(self, *args, **kwargs):
            self.calls += 1
            return {'audio': 'completed.wav'}
    state = {'engine': Engine()}
    previous = weakref.ref(state['engine'])
    monkeypatch.setattr(speech, 'prepare_sample', lambda *args, **kwargs: (np.zeros((1, 1)), tmp_path/'reference.wav', 'digest'))
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'mode': 'clone', 'source': 'source.wav', 'text': 'hello', 'language': 'en'}), encoding='utf-8')
    state['engine'] = worker.run(request, state=state)
    assert state['engine'] is previous()
    assert state['engine'].calls == 1
