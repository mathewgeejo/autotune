from pathlib import Path

import pytest

from studio import recognition, indic
from studio.pronunciation import check_pronunciation


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
    monkeypatch.setattr(recognition, 'full_audit', lambda *args: audit(.25, .04, candidate_speaker))
    class Engine:
        def __init__(self, *args):
            self.precision = {'guidance_scale': 2}
        def synthesize(self, request, folder, progress, prepared=None):
            assert self.guidance_scale == 3
            return {'audio': str(folder/'speech.wav'), 'reference_preview': str(tmp_path/'reference.wav'),
                    'synthesis_seconds': 90, 'precision': dict(self.precision)}
    monkeypatch.setattr(indic, 'IndicEngine', Engine)
    default = baseline(tmp_path)
    result = check_pronunciation({'text': 'expected'}, tmp_path, None, default, lambda message: None, lambda report: None)
    assert result['word_check']['selected'] == selected
    assert result['precision']['guidance_scale'] == (3 if selected == 'retry' else 2)
    assert Path(result['audio']).parent == (tmp_path/'pronunciation-retry' if selected == 'retry' else tmp_path)
    assert result['word_check']['render_synthesis_seconds'] == 170
