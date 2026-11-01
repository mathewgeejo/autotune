"""CPU voice cloning using the publisher's ONNX inference pipeline.

Sessions live in an isolated persistent worker. Speaker conditioning is cached
on disk; generated text is always synthesized, never spliced from the sample.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from studio.audio import AudioError, decode, validate_signal, best_reference
from studio.models import REVISION, PRECISION, download_models
from studio.paths import REFERENCES

RATE = 24000
MAX_TEXT = 5000
START_TOKEN, STOP_TOKEN, SILENCE_TOKEN = 6561, 6562, 4299


def validate_text(text):
    text = (text or '').strip()
    if not text or not any(character.isalnum() for character in text):
        raise AudioError('Type the words you want the voice to say.')
    if len(text) > MAX_TEXT:
        raise AudioError(f'Please use {MAX_TEXT:,} characters or fewer per generation.')
    return text


def split_text(text, limit=240):
    """Bound decoder memory without dropping words or splitting decimal numbers."""
    text = re.sub(r'\s+', ' ', validate_text(text))
    chunks = []
    while len(text) > limit:
        boundaries = list(re.finditer(r'[.!?;][\s]+', text[:limit + 1]))
        cut = boundaries[-1].end() if boundaries else text.rfind(' ', 0, limit + 1)
        if cut <= 0:
            raise AudioError('A word is too long. Add spaces or punctuation to the text.')
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def prepare_sample(source, folder, start=0, end=0):
    start, end = float(start or 0), float(end or 0)
    if not np.isfinite([start, end]).all() or start < 0 or end < 0:
        raise AudioError('Sample times must be finite, positive numbers.')
    if end and end <= start:
        raise AudioError('The sample end must be later than its start.')
    if end and end - start > 60:
        raise AudioError('Select at most 60 seconds for the voice sample.')
    source = Path(source)
    if not source.is_file():
        raise AudioError('Record or upload a voice sample first.')
    if source.stat().st_size > 150 * 1024**2:
        raise AudioError('Upload a voice sample smaller than 150 MB.')
    decoded = decode(source, folder / 'sample-decoded.wav', start=start,
                     duration=end-start if end else 60.1, rate=RATE)
    samples, rate = validate_signal(decoded, minimum=3)
    if len(samples) / rate > 60.02:
        raise AudioError('This sample exceeds 60 seconds. Choose a shorter section below the recording.')
    # Keep the best continuous ten seconds; never pitch-shift or denoise away
    # characteristics that define the reference speaker.
    selected_start, length, _ = best_reference(decoded, seconds=10)
    mono = samples[int(selected_start*rate):int((selected_start+length)*rate), 0]
    # Trim only edge silence, preserving natural pauses inside the sample.
    frame = int(0.02 * rate)
    rms = np.sqrt(np.mean(mono[:len(mono)//frame*frame].reshape(-1, frame)**2, axis=1))
    active = np.flatnonzero(rms > max(0.003, float(rms.max()) * 0.04))
    if not len(active) or len(active)*0.02 < 2:
        raise AudioError('The sample needs at least two seconds of clear, continuous speech.')
    left = max(0, int((active[0]*0.02-0.12)*rate))
    right = min(len(mono), int(((active[-1]+1)*0.02+0.12)*rate))
    mono = mono[left:right]
    peak = float(np.max(np.abs(mono)))
    mono *= min(0.9 / max(peak, 0.001), 3)
    preview = folder / 'voice-sample.wav'
    sf.write(preview, mono, RATE, subtype='PCM_16')
    digest = hashlib.sha256(preview.read_bytes() + REVISION.encode() + b'conditioning-v1').hexdigest()
    return mono[np.newaxis].astype(np.float32), preview, digest


class SpeechEngine:
    def __init__(self, progress):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        paths = download_models(progress)
        self.sessions = {}
        for name in ('speech_encoder', 'embed_tokens', 'language_model', 'conditional_decoder'):
            progress(f'Loading {name.replace("_", " ")} on CPU')
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Sessions don't spin and compete for this laptop's two cores.
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            self.sessions[name] = ort.InferenceSession(paths[name], options,
                                                      providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(paths['tokenizer'])

    def conditioning(self, audio, digest, progress):
        cache = REFERENCES / digest
        path = cache / 'conditioning.npz'
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    values = [saved[key] for key in ('cond', 'prompt', 'embedding', 'features')]
                if not all(value.size and np.isfinite(value).all() for value in values):
                    raise ValueError('Invalid cached conditioning')
                progress('Reusing cached voice sample')
                return values, True
            except (ValueError, OSError, KeyError, EOFError):
                path.unlink(missing_ok=True)
        progress('Learning the tone and characteristics of your voice sample')
        values = self.sessions['speech_encoder'].run(None, {'audio_values': audio})
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / 'conditioning.part.npz'
        np.savez(temporary, **dict(zip(('cond', 'prompt', 'embedding', 'features'), values)))
        temporary.replace(path)
        return values, False

    def generate_chunk(self, text, voice, progress):
        cond, prompt, embedding, features = voice
        language = self.sessions['language_model']
        ids = np.array([self.tokenizer.encode(text).ids], dtype=np.int64)
        embeds = self.sessions['embed_tokens'].run(None, {'input_ids': ids})[0]
        embeds = np.concatenate((cond, embeds), axis=1)
        batch, length, _ = embeds.shape
        past = {}
        for item in language.get_inputs():
            if 'past_key_values' in item.name:
                # Read the exported architecture rather than assuming layer count.
                heads = item.shape[1]
                dimension = item.shape[3]
                dtype = np.float16 if item.type == 'tensor(float16)' else np.float32
                past[item.name] = np.zeros((batch, heads, 0, dimension), dtype=dtype)
        mask = np.ones((batch, length), dtype=np.int64)
        positions = np.arange(length, dtype=np.int64).reshape(1, -1)
        tokens = [START_TOKEN]
        # 25 speech tokens / second, bounded to 40 seconds per text chunk.
        for index in range(1000):
            logits, *present = language.run(None, {'inputs_embeds': embeds,
                'attention_mask': mask, 'position_ids': positions, **past})
            scores = logits[:, -1, :].copy()
            repeated = np.unique(tokens)
            penalized = scores[:, repeated]
            scores[:, repeated] = np.where(penalized < 0, penalized * 1.2, penalized / 1.2)
            token = int(np.argmax(scores, axis=-1)[0])
            if token == STOP_TOKEN:
                break
            tokens.append(token)
            ids = np.array([[token]], dtype=np.int64)
            embeds = self.sessions['embed_tokens'].run(None, {'input_ids': ids})[0]
            mask = np.concatenate((mask, np.ones((batch, 1), dtype=np.int64)), axis=1)
            positions = positions[:, -1:] + 1
            past = dict(zip(past, present))
            if index % 25 == 0:
                progress(f'{index / 25:.1f} seconds of speech composed')
        else:
            raise AudioError('The voice model could not finish this passage. Split it into shorter sentences and retry.')
        if len(tokens) < 3:
            raise AudioError('No speech was generated. Try a clearer sample or a longer sentence.')
        speech_tokens = np.concatenate((prompt, np.array([tokens[1:]], dtype=np.int64),
                                       np.full((batch, 3), SILENCE_TOKEN, dtype=np.int64)), axis=1)
        progress('Rendering the speech waveform')
        wav = self.sessions['conditional_decoder'].run(None, {
            'speech_tokens': speech_tokens, 'speaker_embeddings': embedding,
            'speaker_features': features})[0].reshape(-1)
        if not len(wav) or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 0.001:
            raise AudioError('The voice model returned empty audio. Try another clear speech sample.')
        return wav

    def synthesize(self, request, folder, progress, prepared=None):
        started = time.monotonic()
        chunks = split_text(request.get('text'))
        progress('Preparing the voice sample')
        sample, preview, digest = prepared or prepare_sample(
            request['source'], folder, request.get('start'), request.get('end'))
        voice, reused = self.conditioning(sample, digest, progress)
        audio_path = folder / 'speech.wav'
        first_audio = None
        with sf.SoundFile(audio_path, 'w', samplerate=RATE, channels=1, subtype='PCM_16') as output:
            for index, text in enumerate(chunks):
                def report(message):
                    progress(f'Passage {index+1}/{len(chunks)}: {message}')
                wav = self.generate_chunk(text, voice, report)
                wav *= min(1, 0.98 / max(float(np.max(np.abs(wav))), 0.001))
                output.write(wav)
                if first_audio is None:
                    first_audio = time.monotonic() - started
                # A closed, immutable preview is safe to play while the final WAV
                # is still being written. Never serve the unfinished export.
                part = folder / f'passage-{index+1}.wav'
                sf.write(part, wav, RATE, subtype='PCM_16')
                if index < len(chunks) - 1:
                    progress(f'Passage {index+1}/{len(chunks)} ready', preview=str(part))
        result = {'audio': str(audio_path), 'reference_preview': str(preview),
                  'duration': sf.info(audio_path).duration, 'passages': len(chunks),
                  'cached_voice': reused, 'first_audio_seconds': first_audio,
                  'synthesis_seconds': time.monotonic()-started, 'model_revision': REVISION, 'precision': PRECISION}
        (folder / 'speech-metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
