"""Persistent DhVaani/ZipVoice inference for native Indic scripts.

Graph and DSP contract: Bbkblo/DhVaani-0.5-ONNX at the pinned model revision.
The exporter publishes this contract under Apache-2.0. See THIRD_PARTY.md.
"""
import hashlib
import os
import unicodedata

import numpy as np

from studio.audio import AudioError
from studio.models import download_indic_models, INDIC_REVISION, INDIC_ID
from studio.paths import REFERENCES
from studio.speech import SpeechEngine


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('VOICE_STUDIO_THREADS', '2'))
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def mel_features(audio, filters, window):
    """Exact exported HTK filterbank, rather than an approximate substitute."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    padded = np.pad(audio, (512, 512))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 1024)[::256]
    magnitude = np.abs(np.fft.rfft(frames*window, axis=1)).astype(np.float32)
    result = np.log(np.maximum(magnitude @ filters, 1e-7))
    target = (len(audio)+128)//256
    if len(result) < target:
        result = np.pad(result, ((0, target-len(result)), (0, 0)), mode='edge')
    return result[:target].astype(np.float32)


def overlap_add(spectrum, window, hop):
    size = len(window)
    frames = np.fft.irfft(spectrum, n=size, axis=1)*window[None, :, None]
    count = frames.shape[2]
    output_size = (count-1)*hop+size
    output = np.zeros((len(frames), output_size), dtype=np.float32)
    envelope = np.zeros(output_size, dtype=np.float32)
    for index in range(count):
        section = slice(index*hop, index*hop+size)
        output[:, section] += frames[:, :, index]
        envelope[section] += window*window
    pad = (size-hop)//2
    return output[:, pad:output_size-pad]/np.maximum(envelope[None, pad:output_size-pad], 1e-11)


class IndicEngine(SpeechEngine):
    revision = INDIC_REVISION
    model_id = INDIC_ID

    def __init__(self, progress, quality='Detailed'):
        if quality not in ('Balanced', 'Detailed'):
            raise AudioError('Choose Balanced or Detailed Indic quality.')
        self.quality = quality
        self.steps = 32 if quality == 'Detailed' else 16
        suffix = '' if quality == 'Detailed' else '_int8'
        self.precision = {'text_encoder': 'fp32' if not suffix else 'int8',
                          'flow_decoder': 'fp32' if not suffix else 'int8', 'steps': self.steps}
        root = download_indic_models(quality, progress)
        progress('Loading native-script text and voice models')
        self.text_encoder = session(root/f'text_encoder{suffix}.onnx')
        self.flow_decoder = session(root/f'fm_decoder{suffix}.onnx')
        self.vocoder = session(root/'vocoder_backbone.onnx')
        with np.load(root/'mel_fb.npz', allow_pickle=False) as data:
            self.filters, self.window = data['fb'].astype(np.float32), data['window'].astype(np.float32)
        with np.load(root/'vocos_head.npz', allow_pickle=False) as data:
            self.vocoder_weight = data['linear_weight'].astype(np.float32)
            self.vocoder_bias = data['linear_bias'].astype(np.float32)
            self.vocoder_window = data['window'].astype(np.float32)
            self.vocoder_hop = int(data['hop_length'])
        self.tokens = {}
        for line in (root/'tokens.txt').read_text(encoding='utf-8').splitlines():
            character, number = line.rsplit('\t', 1)
            self.tokens[character] = int(number)
        self.reference_text = ''

    def text_ids(self, text):
        text = unicodedata.normalize('NFC', text).strip()
        text = text.translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"', '।': '.', '…': '.'}))
        normalized = []
        missing = set()
        for char in text:
            if char in self.tokens:
                normalized.append(char)
            elif char.isspace():
                normalized.append(' ')
            elif unicodedata.category(char).startswith('P'):
                normalized.append(' ')
            elif char in ('\u200c', '\u200d', '\ufeff'):
                continue
            else:
                missing.add(char)
        if missing:
            raise AudioError('Unsupported characters: '+', '.join(sorted(missing))+'. Spell out symbols or numbers in words.')
        text = ''.join(normalized).strip()
        if not text:
            raise AudioError('Type a sentence containing supported letters.')
        if text[-1] not in '.!?;:,':
            text += '.'
        return np.array([[self.tokens[char] for char in text if char in self.tokens]], dtype=np.int64)

    def conditioning(self, audio, digest, progress):
        digest = hashlib.sha256((digest+INDIC_REVISION).encode()).hexdigest()
        cache = REFERENCES/digest
        path = cache/'indic-conditioning.npz'
        reused = False
        try:
            with np.load(path, allow_pickle=False) as data:
                features, original_rms = data['features'], float(data['rms'])
            if features.ndim != 3 or features.shape[2] != 100 or not np.isfinite(features).all():
                raise ValueError('Invalid cached reference features')
            reused = True
            progress('Reusing cached Indic voice conditioning')
        except (OSError, ValueError, KeyError, EOFError):
            progress('Encoding the selected reference voice')
            original_rms = float(np.sqrt(np.mean(audio*audio)))
            normalized = audio*max(1, 0.1/max(original_rms, 1e-8))
            features = mel_features(normalized, self.filters, self.window)[None]*0.1
            cache.mkdir(parents=True, exist_ok=True)
            temporary = cache/'indic-conditioning.part.npz'
            np.savez(temporary, features=features, rms=np.array(original_rms))
            temporary.replace(path)
        return (features.astype(np.float32), self.text_ids(self.reference_text), original_rms), reused

    def generate_chunk(self, text, voice, progress):
        features, prompt_tokens, original_rms = voice
        prompt_length = features.shape[1]
        text_condition = self.text_encoder.run(None, {
            'tokens': self.text_ids(text), 'prompt_tokens': prompt_tokens,
            'prompt_features_len': np.array(prompt_length, dtype=np.int64),
            'speed': np.array(1.0, dtype=np.float32)})[0].astype(np.float32)
        batch, length, dimension = text_condition.shape
        if length <= prompt_length or length > 3000:
            raise AudioError('The estimated passage length is invalid. Check the reference transcript or use shorter text.')
        noise = np.random.RandomState(666).randn(batch, length, dimension).astype(np.float32)
        conditioning = np.zeros_like(noise)
        conditioning[:, :prompt_length] = features
        schedule = np.linspace(0, 1, self.steps+1, dtype=np.float32)
        schedule = 0.5*schedule/(1-0.5*schedule)
        for step in range(self.steps):
            progress(f'Refining pronunciation and tone: {step+1}/{self.steps}')
            velocity = self.flow_decoder.run(None, {'t': np.array(schedule[step], dtype=np.float32),
                'x': noise, 'text_condition': text_condition, 'speech_condition': conditioning,
                'guidance_scale': np.array(1.0, dtype=np.float32)})[0]
            noise += velocity*(schedule[step+1]-schedule[step])
        mel = noise[:, prompt_length:].transpose(0, 2, 1)/0.1
        progress('Rendering the speech waveform')
        vocoder_input = self.vocoder.get_inputs()[0].name
        hidden = self.vocoder.run(None, {vocoder_input: mel.astype(np.float32)})[0]
        head = (hidden @ self.vocoder_weight.T + self.vocoder_bias).transpose(0, 2, 1)
        magnitude, phase = np.split(head, 2, axis=1)
        magnitude = np.exp(np.minimum(magnitude, np.log(100)))
        spectrum = magnitude*(np.cos(phase)+1j*np.sin(phase))
        audio = overlap_add(spectrum, self.vocoder_window, self.vocoder_hop)[0]
        if original_rms < 0.1:
            audio *= original_rms/0.1
        if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
            raise AudioError('The Indic model returned empty audio. Check the reference and transcript.')
        return audio.astype(np.float32)

    def synthesize(self, request, folder, progress, prepared=None):
        self.reference_text = request.get('reference_text', '').strip()
        if not self.reference_text:
            raise AudioError('The selected reference needs a transcript. Use automatic transcription or review its words.')
        result = super().synthesize(request, folder, progress, prepared)
        result['reference_text'] = self.reference_text
        return result
