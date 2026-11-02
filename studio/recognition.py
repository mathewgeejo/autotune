"""Independent, multilingual content and speaker checks; lazy model loading."""
import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from studio.audio import decode
from studio.models import download_asr, download_speaker, ASR_ID, ASR_REVISION, SPEAKER_ID, SPEAKER_REVISION


def evaluation_level(audio):
    """Match listening level without changing timing, pitch, or word content."""
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio*audio)))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    gain = min(0.08/max(rms, 1e-9), 0.95/max(peak, 1e-9)) if peak else 1
    return audio*gain, {'method': 'RMS 0.08, peak-limited at 0.95; no dynamic compression',
                        'original_rms': rms, 'gain_db': float(20*np.log10(max(gain, 1e-9)))}


def recognize(path, folder, language=None, progress=lambda message: None):
    from faster_whisper import WhisperModel
    from studio.paths import MODELS
    root = download_asr(progress)
    progress('Recognizing the spoken words on CPU')
    model = WhisperModel(root, device='cpu', compute_type='int8', cpu_threads=2, num_workers=1,
                         download_root=str(MODELS), local_files_only=True)
    audio_path = decode(path, Path(folder)/'recognition-input.wav', rate=16000, duration=60.1)
    audio, _ = sf.read(audio_path, dtype='float32')
    audio, level = evaluation_level(audio)
    # No expected text is passed as an initial prompt: doing so would bias the
    # recognizer toward the answer and invalidate a pronunciation/content test.
    segments, info = model.transcribe(audio, language=language or None, beam_size=5,
        vad_filter=True, word_timestamps=True, condition_on_previous_text=False)
    collected = []
    for segment in segments:
        collected.append({'start': segment.start, 'end': segment.end, 'text': segment.text.strip(),
                          'average_log_probability': segment.avg_logprob,
                          'words': [{'start': word.start, 'end': word.end, 'word': word.word,
                                     'probability': word.probability} for word in segment.words or []]})
        progress(f'Recognized through {segment.end:.1f} seconds')
    result = {'text': ' '.join(segment['text'] for segment in collected), 'segments': collected,
              'language': info.language, 'language_probability': info.language_probability,
              'model': ASR_ID, 'revision': ASR_REVISION, 'input_preprocessing': level}
    del model
    gc.collect()
    return result


def speaker_similarity(reference, candidate, folder, progress=lambda message: None):
    import onnxruntime as ort
    graph = download_speaker(progress)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    model = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
    preprocessing = json.loads((graph.parent.parent/'preprocessor_config.json').read_text(encoding='utf-8'))
    embeddings = []
    for name, path in [('reference', reference), ('output', candidate)]:
        progress(f'Checking independent speaker identity: {name}')
        decoded = decode(path, Path(folder)/f'verifier-{name}.wav', rate=16000, duration=60)
        audio, _ = sf.read(decoded, dtype='float32')
        audio, _ = evaluation_level(audio)
        vectors = []
        # Several bounded windows use the whole minute without allocating a
        # minute-long attention tensor. Silence-only windows contribute nothing.
        for start in range(0, len(audio), 8*16000):
            clip = audio[start:start+8*16000]
            if len(clip) < 16000 or np.sqrt(np.mean(clip*clip)) < 0.003:
                continue
            if preprocessing.get('do_normalize', False):
                clip = (clip-clip.mean()) / np.sqrt(clip.var()+1e-7)
            feeds = {}
            for item in model.get_inputs():
                if item.name == 'input_values':
                    feeds[item.name] = clip[None].astype(np.float32)
                elif item.name == 'attention_mask':
                    feeds[item.name] = np.ones((1, len(clip)), dtype=np.int64)
                else:
                    raise ValueError(f'Unsupported speaker-verifier input: {item.name}')
            outputs = dict(zip((item.name for item in model.get_outputs()), model.run(None, feeds)))
            if 'embeddings' not in outputs:
                raise ValueError('Speaker model did not return embeddings; no substitute score was computed.')
            embedding = outputs['embeddings'].reshape(-1)
            vectors.append(embedding/max(float(np.linalg.norm(embedding)), 1e-9))
        if not vectors:
            return {'available': False, 'reason': f'Not enough clear speech in the {name} clip.'}
        mean = np.mean(vectors, axis=0)
        embeddings.append(mean/max(float(np.linalg.norm(mean)), 1e-9))
    score = float(np.clip(np.dot(*embeddings), -1, 1))
    return {'available': True, 'cosine_similarity': score, 'model': SPEAKER_ID,
            'revision': SPEAKER_REVISION, 'independent_of_synthesizer': True,
            'input_preprocessing': 'RMS level-matched before the published model feature-extractor settings',
            'interpretation': 'Higher means more similar to this verifier. It is not a percentage of identity, '
                              'and no universal indistinguishability threshold is claimed.'}


def full_audit(reference, output, expected, folder, language, progress):
    from studio.analysis import transcript_errors
    verifier = speaker_similarity(reference, output, folder, progress)
    # Speaker sessions are released on function exit before the ASR model loads.
    gc.collect()
    transcript = recognize(output, folder, language, progress)
    errors = transcript_errors(expected, transcript['text'])
    if errors.get('available'):
        words = [word for segment in transcript['segments'] for word in segment['words']]
        for difference in errors['differences']:
            index = difference['recognized_word_index']
            if words:
                difference['output_seconds'] = words[min(index, len(words)-1)]['start']
    result = {'speaker_verification': verifier, 'transcription': transcript, 'content_comparison': errors}
    Path(folder, 'quality-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result
