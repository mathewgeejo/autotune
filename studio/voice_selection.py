"""Compare readings with the same judges, preserving the default voice as anchor.

These engineering guards support a pronunciation retry. They do not establish
human indistinguishability or interpret speaker cosine as a percentage.
"""
import math

MAX_SPEAKER_DROP = 0.03


def _numbers(audit):
    content = audit.get('content_comparison', {})
    speaker = audit.get('speaker_verification', {})
    if not content.get('available') or not speaker.get('available'):
        return None
    try:
        values = (float(content['word_error_rate']), float(content['grapheme_error_rate']),
                  float(speaker['cosine_similarity']))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values) or min(values[:2]) < 0 or not -1 <= values[2] <= 1:
        return None
    return values


def compare_reading(default, candidate):
    """Require content improvement and bounded speaker-score loss vs the default."""
    base, other = _numbers(default), _numbers(candidate)
    result = {'accepted': False, 'max_speaker_cosine_drop': MAX_SPEAKER_DROP,
              'scope': 'Relative automated guard for the same text/reference; not a perceptual pass.'}
    if base is None or other is None:
        return {**result, 'reason': 'Complete finite content and speaker checks are required for both readings.'}
    for section in ('transcription', 'speaker_verification'):
        for key in ('model', 'revision'):
            if not default.get(section, {}).get(key) or default[section].get(key) != candidate.get(section, {}).get(key):
                return {**result, 'reason': 'Different or missing judge metadata makes the readings incomparable.'}
    result.update(default_word_error=base[0], candidate_word_error=other[0],
                  default_grapheme_error=base[1], candidate_grapheme_error=other[1],
                  default_speaker_cosine=base[2], candidate_speaker_cosine=other[2])
    if other[0] > base[0] or other[1] > base[1] or other[:2] == base[:2]:
        return {**result, 'reason': 'Word/grapheme errors must improve without worsening the other content score.'}
    if other[2] < base[2]-MAX_SPEAKER_DROP:
        return {**result, 'reason': 'The candidate loses too much speaker similarity against the default anchor.'}
    return {**result, 'accepted': True, 'reason': 'Content improves and the same-reference speaker score stays within the relative guard.'}
