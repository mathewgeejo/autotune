from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from studio import recognition, indic
from studio.pronunciation import check_pronunciation


def test_english_retry_uses_shorter_passages_and_releases_model_before_audit(tmp_path, monkeypatch):
    from studio import speech
    original = audit(.5, .2, .9)
    monkeypatch.setattr(recognition, 'recognize', lambda path, folder, language, progress:
                        original['transcription'] if language == 'en' else pytest.fail('English ASR required'))
    monkeypatch.setattr(recognition, 'content_comparison', lambda *args: original['content_comparison'])
    monkeypatch.setattr(recognition, 'speaker_similarity', lambda *args: original['speaker_verification'])
    references = []
    class Engine:
        def __init__(self, progress):
            self.precision = {'language_model': 'q4'}
            references.append(weakref.ref(self))
        def synthesize(self, request, folder, progress, prepared=None):
            assert self.english_max_words == 8 and self.english_max_chars == 80
            return {'audio': str(folder/'speech.wav'), 'reference_preview': 'reference.wav',
                    'synthesis_seconds': 30, 'precision': dict(self.precision)}
    monkeypatch.setattr(speech, 'SpeechEngine', Engine)
    def candidate(*args):
        assert references[0]() is None
        assert args[4] == 'en'
        return audit(0, 0, .91)
    monkeypatch.setattr(recognition, 'full_audit', candidate)
    result = check_pronunciation({'text': 'Complete English words.', 'language': 'en'}, tmp_path, None,
                                baseline(tmp_path), lambda message: None, lambda report: None)
    assert result['word_check']['selected'] == 'retry'


def test_long_word_check_recognizes_all_passages_with_global_timestamps(tmp_path, monkeypatch):
    import soundfile as sf
    from studio.pronunciation import reading_transcript
    paths = []
    def recognize(path, folder, language, progress):
        paths.append(path.name)
        return {'text': path.stem, 'model': 'recognizer', 'revision': 'fixed',
                'script_check': {'reliable': True}, 'segments': [{'start': 1., 'end': 2., 'words': []}]}
    monkeypatch.setattr(recognition, 'recognize', recognize)
    monkeypatch.setattr(sf, 'info', lambda path: SimpleNamespace(duration=35.))
    result = reading_transcript({'audio': str(tmp_path/'speech.wav'), 'duration': 70, 'passages': 2},
                                tmp_path/'checks', 'en', lambda message: None)
    assert paths == ['passage-1.wav', 'passage-2.wav']
    assert result['text'] == 'passage-1 passage-2'
    assert result['segments'][1]['start'] == 36.
    assert result['checked_passages'] == 2


def audit(word, grapheme, speaker):
    return {'transcription': {'model': 'recognizer', 'revision': 'fixed'},
            'content_comparison': {'available': True, 'word_error_rate': word, 'grapheme_error_rate': grapheme},
            'speaker_verification': {'available': True, 'cosine_similarity': speaker,
                                     'model': 'verifier', 'revision': 'fixed'}}


def baseline(tmp_path):
    return {'audio': str(tmp_path/'speech.wav'), 'reference_preview': str(tmp_path/'reference.wav'),
            'synthesis_seconds': 80, 'precision': {'guidance_scale': 2}}


def test_recognizer_failure_preserves_a_completed_synthesis(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError('Offline recognizer download unavailable')
    monkeypatch.setattr(recognition, 'recognize', failed)
    default = baseline(tmp_path)
    records = []
    result = check_pronunciation({'text': 'expected'}, tmp_path, None, default, lambda message: None, records.append)
    assert result['audio'] == default['audio']
    assert result['word_check']['status'] == 'check-unavailable'
    assert 'word_check' not in default
    assert records[0]['selected'] == 'default'


def test_correct_words_skip_retry_and_speaker_models(tmp_path, monkeypatch):
    monkeypatch.setattr(recognition, 'recognize', lambda *args: {'text': 'expected'})
    monkeypatch.setattr(recognition, 'content_comparison', lambda *args: audit(0, 0, .9)['content_comparison'])
    def unexpected(*args, **kwargs):
        pytest.fail('No retry or speaker model should load when words already match')
    monkeypatch.setattr(recognition, 'speaker_similarity', unexpected)
    monkeypatch.setattr(indic, 'IndicEngine', unexpected)
    result = check_pronunciation({'text': 'expected'}, tmp_path, None, baseline(tmp_path), lambda message: None, lambda report: None)
    assert result['word_check']['status'] == 'word-check-passed'
    assert not result['word_check']['retry_attempted']


@pytest.mark.parametrize('candidate_speaker,selected', [(.955, 'retry'), (.686, 'default')])
def test_retry_requires_word_gain_and_retains_actual_selected_metadata(tmp_path, monkeypatch, candidate_speaker, selected):
    original = audit(.5, .09, .975)
    monkeypatch.setattr(recognition, 'recognize', lambda *args: original['transcription'])
    monkeypatch.setattr(recognition, 'content_comparison', lambda *args: original['content_comparison'])
    monkeypatch.setattr(recognition, 'speaker_similarity', lambda *args: original['speaker_verification'])
    flow_refs = []
    def retry_check(*args):
        assert flow_refs[0]() is None, 'Warm-up wrapper must not retain the GPU model during checking'
        return audit(.25, .04, candidate_speaker)
    monkeypatch.setattr(recognition, 'full_audit', retry_check)
    class Engine:
        def __init__(self, *args):
            self.precision = {'guidance_scale': 2}
            self.flow_decoder = SimpleNamespace(backend='openvino-intel-gpu-fp32')
        def synthesize(self, request, folder, progress, prepared=None):
            assert self.guidance_scale == 3
            assert self.flow_decoder.evaluations == 3
            flow_refs.append(weakref.ref(self.flow_decoder))
            return {'audio': str(folder/'speech.wav'), 'reference_preview': str(tmp_path/'reference.wav'),
                    'synthesis_seconds': 90, 'precision': dict(self.precision)}
    monkeypatch.setattr(indic, 'IndicEngine', Engine)
    default = baseline(tmp_path)
    result = check_pronunciation({'text': 'expected'}, tmp_path, None, default, lambda message: None, lambda report: None)
    assert result['word_check']['selected'] == selected
    assert result['precision']['guidance_scale'] == (3 if selected == 'retry' else 2)
    assert Path(result['audio']).parent == (tmp_path/'pronunciation-retry' if selected == 'retry' else tmp_path)
    assert result['word_check']['render_synthesis_seconds'] == 170
    assert result['word_check']['candidate_precision']['gpu_shape_warmup_evaluations'] == 3


@pytest.mark.parametrize('failure_stage', ['render', 'retry-check'])
def test_failed_retry_keeps_the_default_and_releases_its_engine(tmp_path, monkeypatch, failure_stage):
    original = audit(.5, .09, .975)
    monkeypatch.setattr(recognition, 'recognize', lambda *args: original['transcription'])
    monkeypatch.setattr(recognition, 'content_comparison', lambda *args: original['content_comparison'])
    monkeypatch.setattr(recognition, 'speaker_similarity', lambda *args: original['speaker_verification'])
    refs = []
    class Engine:
        def __init__(self, *args):
            self.precision = {}
            self.flow_decoder = object()
            refs.append(weakref.ref(self))
        def synthesize(self, request, folder, progress, prepared=None):
            if failure_stage == 'render':
                raise MemoryError('Simulated retry memory failure')
            return {'audio': str(folder/'speech.wav'), 'synthesis_seconds': 90}
    def failed_check(*args):
        assert refs[0]() is None, 'The retry model must be released before its checker loads'
        raise OSError('Simulated checker failure')
    monkeypatch.setattr(indic, 'IndicEngine', Engine)
    monkeypatch.setattr(recognition, 'full_audit', failed_check)
    default = baseline(tmp_path)
    result = check_pronunciation({'text': 'expected'}, tmp_path, None, default, lambda message: None, lambda report: None)
    assert result['audio'] == default['audio']
    assert result['word_check']['status'] == 'check-unavailable'
    assert result['word_check']['retry_attempted']
    assert refs[0]() is None


def test_separate_report_write_failure_keeps_audio_and_embedded_check(tmp_path, monkeypatch):
    monkeypatch.setattr(recognition, 'recognize', lambda *args: {'text': 'expected'})
    monkeypatch.setattr(recognition, 'content_comparison', lambda *args: audit(0, 0, .9)['content_comparison'])
    def failed_write(report):
        raise PermissionError('Report temporarily locked')
    result = check_pronunciation({'text': 'expected'}, tmp_path, None, baseline(tmp_path), lambda message: None, failed_write)
    assert result['audio'] == baseline(tmp_path)['audio']
    assert result['word_check']['status'] == 'word-check-passed'
