import pytest

from studio.audio import AudioError
from studio.languages import resolve_language


def test_native_script_routing_never_falls_back_to_english():
    assert resolve_language('Hello from Kerala.') == 'en'
    assert resolve_language('नमस्ते, आप कैसे हैं?') == 'hi'
    assert resolve_language('നമസ്കാരം, സുഖമാണോ?') == 'ml'
    assert resolve_language('നാളെ meeting ഉണ്ട്.') == 'ml'
    with pytest.raises(AudioError, match='Indic script'):
        resolve_language('नमस्ते', 'en')
    with pytest.raises(AudioError, match='does not match'):
        resolve_language('നമസ്കാരം', 'hi')
