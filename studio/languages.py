"""Explicit language routing: never send Indic script to an English model."""
from studio.audio import AudioError

LANGUAGES = {'en': 'English', 'hi': 'हिन्दी · Hindi', 'ml': 'മലയാളം · Malayalam'}


def resolve_language(text, requested='auto'):
    malayalam = sum('\u0d00' <= ch <= '\u0d7f' for ch in text)
    hindi = sum('\u0900' <= ch <= '\u097f' for ch in text)
    detected = 'ml' if malayalam > hindi else 'hi' if hindi else 'en'
    if requested == 'auto':
        return detected
    if requested not in LANGUAGES:
        raise AudioError('Choose English, Hindi, or Malayalam.')
    if requested == 'en' and (malayalam or hindi):
        raise AudioError('Choose Auto, Hindi, or Malayalam for text written in an Indic script.')
    if requested == 'hi' and malayalam or requested == 'ml' and hindi:
        raise AudioError('The selected language does not match the text script.')
    return requested
