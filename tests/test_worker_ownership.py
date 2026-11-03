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
    request.write_text(json.dumps({'mode': 'clone', 'source': 'source.wav', 'text': 'hello', 'language': 'en',
                                  'reference_text': 'known words'}), encoding='utf-8')
    state['engine'] = worker.run(request, state=state)
    assert state['engine'] is previous()
    assert state['engine'].calls == 1


def test_english_blank_reference_is_transcribed_after_releasing_warm_model(tmp_path, monkeypatch):
    class Engine:
        engine_key = ('en',)
        def synthesize(self, request, *args, **kwargs):
            assert request['reference_text'] == 'Selected clip words.'
            assert request['reference_full_text'] == 'The complete recording words.'
            return {'audio': 'completed.wav', 'reference_text': request['reference_text']}
    state = {'engine': Engine()}
    previous = weakref.ref(state['engine'])
    monkeypatch.setattr(speech, 'prepare_sample', lambda *args, **kwargs: (None, tmp_path/'clip.wav', 'digest'))
    def transcribe(*args):
        assert previous() is None
        return {'reference_text': 'Selected clip words.', 'reference_full_text': 'The complete recording words.'}
    monkeypatch.setattr(worker, 'transcribe_reference', transcribe)
    monkeypatch.setattr(speech, 'SpeechEngine', lambda progress: Engine())
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'mode': 'clone', 'source': 'source.wav', 'text': 'Hello.', 'language': 'en'}))
    assert worker.run(request, state=state) is not None
    assert (tmp_path/'result.json').is_file()
