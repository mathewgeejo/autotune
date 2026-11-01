import numpy as np
import pytest
import soundfile as sf

from studio.audio import AudioError
from studio.speech import prepare_sample, split_text, SpeechEngine, STOP_TOKEN


def test_long_text_keeps_every_word_and_numbers_in_order():
    text = ('In 2026 the price was 12.75 dollars. We read each sentence completely! ' * 30).strip()
    chunks = split_text(text)
    assert len(chunks) > 1
    assert ' '.join(chunks) == text
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert all('12.' not in chunk or '12.75' in chunk for chunk in chunks)
    with pytest.raises(AudioError, match='words'):
        split_text(' ... ')
    with pytest.raises(AudioError, match='5,000'):
        split_text('hello ' * 1000)


def test_sample_selection_bounds_silence_and_cache_identity(tmp_path):
    rate = 24000
    t = np.arange(14 * rate) / rate
    # Vary the tone's loudness to check long-reference selection and padding.
    signal = 0.2 * np.sin(2*np.pi*180*t) * (0.8 + 0.2*np.sin(2*np.pi*3*t))
    source = tmp_path / 'voice.wav'
    sf.write(source, np.r_[np.zeros(5*rate), signal], rate)
    audio, preview, digest = prepare_sample(source, tmp_path)
    assert 3 * rate <= audio.shape[1] <= 10 * rate + 1
    assert audio.shape[0] == 1
    assert np.isfinite(audio).all()
    assert sf.info(preview).samplerate == rate
    assert prepare_sample(source, tmp_path)[2] == digest
    assert prepare_sample(source, tmp_path, 7, 12)[2] != digest
    with pytest.raises(AudioError, match='finite'):
        prepare_sample(source, tmp_path, float('nan'))
    sf.write(source, np.zeros(5*rate), rate)
    with pytest.raises(AudioError, match='silent'):
        prepare_sample(source, tmp_path)


def test_stop_token_does_not_discard_the_last_spoken_token():
    """The published example slices [1:-1]; our loop excludes STOP itself."""
    class Tokenizer:
        def encode(self, text):
            return type('Encoded', (), {'ids': [2, 3]})()

    class Embedding:
        def run(self, _, inputs):
            return [np.zeros((*inputs['input_ids'].shape, 4), dtype=np.float32)]

    class Language:
        def __init__(self):
            self.index = 0
        def get_inputs(self):
            return []
        def run(self, _, inputs):
            token = (14, 15, STOP_TOKEN)[self.index]
            self.index += 1
            logits = np.zeros((1, 1, 6563), dtype=np.float32)
            logits[0, 0, token] = 10
            return [logits]

    class Decoder:
        def run(self, _, inputs):
            assert inputs['speech_tokens'].tolist() == [[7, 14, 15, 4299, 4299, 4299]]
            return [np.ones((1, 24000), dtype=np.float32) * 0.1]

    engine = SpeechEngine.__new__(SpeechEngine)
    engine.tokenizer = Tokenizer()
    engine.sessions = {'embed_tokens': Embedding(), 'language_model': Language(),
                       'conditional_decoder': Decoder()}
    voice = [np.zeros((1, 2, 4), dtype=np.float32), np.array([[7]]), None, None]
    assert len(engine.generate_chunk('Hello world.', voice, lambda text: None)) == 24000


def test_reference_limits_and_single_transient_are_rejected(tmp_path):
    path = tmp_path / 'long.wav'
    rate = 24000
    sf.write(path, 0.1*np.sin(2*np.pi*220*np.arange(61*rate)/rate), rate)
    with pytest.raises(AudioError, match='exceeds 60'):
        prepare_sample(path, tmp_path)
    with pytest.raises(AudioError, match='later'):
        prepare_sample(path, tmp_path, 8, 4)
    sf.write(path, np.r_[np.zeros(2*rate), np.ones(rate//5)*0.5, np.zeros(rate)], rate)
    with pytest.raises(AudioError, match='continuous speech'):
        prepare_sample(path, tmp_path)
