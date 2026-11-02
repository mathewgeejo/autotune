import copy

import pytest

from studio.voice_selection import compare_reading


def audit(word_error, grapheme_error, speaker):
    return {'content_comparison': {'available': True, 'word_error_rate': word_error,
                                  'grapheme_error_rate': grapheme_error},
            'speaker_verification': {'available': True, 'cosine_similarity': speaker,
                                     'model': 'verifier', 'revision': 'fixed'},
            'transcription': {'model': 'recognizer', 'revision': 'fixed'}}


def test_hard_sentence_improvement_retains_the_default_voice():
    assert compare_reading(audit(.5, .091, .9749), audit(.25, .0455, .9553))['accepted']


def test_word_gain_cannot_override_the_observed_reconstruction_voice_loss():
    result = compare_reading(audit(.3, .0816, .9694), audit(.2, .0612, .6865))
    assert not result['accepted']
    assert 'speaker' in result['reason']


@pytest.mark.parametrize('candidate', [audit(0, 0, .99), audit(.25, .12, .99), audit(.6, .01, .99)])
def test_no_gain_or_a_content_regression_keeps_the_default(candidate):
    baseline = audit(0, 0, .92) if candidate['content_comparison']['word_error_rate'] == 0 else audit(.5, .09, .97)
    assert not compare_reading(baseline, candidate)['accepted']


def test_unavailable_or_nonfinite_checks_and_different_judges_are_not_accepted():
    baseline = audit(.5, .09, .97)
    for problem in ('missing', 'nonfinite', 'different'):
        candidate = audit(.25, .04, .96)
        if problem == 'missing':
            candidate['speaker_verification']['available'] = False
        elif problem == 'nonfinite':
            candidate['speaker_verification']['cosine_similarity'] = float('nan')
        else:
            candidate['transcription']['revision'] = 'other'
        before = copy.deepcopy(baseline)
        assert not compare_reading(baseline, candidate)['accepted']
        assert baseline == before


@pytest.mark.parametrize('candidate_hash,accepted', [('a'*64, True), ('b'*64, False), (None, False)])
def test_retry_cannot_mix_storage_checkpoints_under_the_same_source_revision(candidate_hash, accepted):
    baseline, candidate = audit(.5, .09, .97), audit(.25, .04, .96)
    baseline['transcription']['checkpoint'] = {'weights_sha256': 'a'*64}
    if candidate_hash:
        candidate['transcription']['checkpoint'] = {'weights_sha256': candidate_hash}
    assert compare_reading(baseline, candidate)['accepted'] is accepted
