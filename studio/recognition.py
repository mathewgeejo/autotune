"""Independent, multilingual content and speaker checks; lazy model loading."""
import gc
import json
from pathlib import Path
import unicodedata

import numpy as np
import soundfile as sf

from studio.audio import decode, reference_activity
from studio.models import (download_asr, download_speaker, ASR_ID, ASR_REVISION, SPEAKER_ID,
                           SPEAKER_REVISION, MALAYALAM_ASR_ID, MALAYALAM_ASR_REVISION)
from studio.runtime import active_time


def transcript_script_check(text, language):
    """Reject an incompatible written script, without using the expected words."""
    letters = [char for char in text if unicodedata.category(char).startswith('L')]
    if not letters:
        return {'reliable': False, 'reason': 'No spoken words were recognized.'}
    ranges = {'ml': ('\u0d00', '\u0d7f'), 'hi': ('\u0900', '\u097f')}
    if language == 'en':
        fraction = sum('LATIN' in unicodedata.name(char, '') for char in letters)/len(letters)
        return {'reliable': fraction >= 0.5, 'native_letter_fraction': fraction,
                'reason': ('Recognized script matches English.' if fraction >= 0.5 else
                           'The recognizer returned a different script. Review the audio and transcript.')}
    if language not in ranges:
        return {'reliable': False, 'reason': 'The detected language is outside English, Hindi, and Malayalam. Select the spoken language explicitly.'}
    lower, upper = ranges[language]
    matching = sum(lower <= char <= upper for char in letters)
    fraction = matching/len(letters)
    reliable = fraction >= 0.5
    return {'reliable': reliable, 'native_letter_fraction': fraction,
            'reason': ('Recognized script matches the selected language.' if reliable else
                       'The recognizer returned a different script. Word-error scores are unavailable; review the audio and transcript.')}


def content_comparison(expected, transcript):
    from studio.analysis import transcript_errors
    check = transcript.get('script_check', {'reliable': False, 'reason': 'Transcript script was not validated.'})
    if not check['reliable']:
        return {'available': False, 'reason': check['reason'], 'differences': []}
    return transcript_errors(expected, transcript['text'])


def evaluation_level(audio):
    """Match listening level without changing timing, pitch, or word content."""
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio*audio)))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    gain = min(0.08/max(rms, 1e-9), 0.95/max(peak, 1e-9)) if peak else 1
    return audio*gain, {'method': 'RMS 0.08, peak-limited at 0.95; no dynamic compression',
                        'original_rms': rms, 'gain_db': float(20*np.log10(max(gain, 1e-9)))}


def verifier_input(audio, rate=16000):
    """A boundary click must not determine the verifier's listening level."""
    audio = np.asarray(audio, dtype=np.float32)
    start, end = 0, len(audio)
    removed = []
    if len(audio) >= rate:
        activity, removed = reference_activity(audio, rate)
        active = np.flatnonzero(activity)
        if removed and len(active):
            if any(item['start_seconds'] < 0.04 for item in removed):
                start = max(0, int((active[0]*0.02-0.12)*rate))
            if any(item['end_seconds'] > len(audio)/rate-0.06 for item in removed):
                end = min(len(audio), int(((active[-1]+1)*0.02+0.12)*rate))
    adjusted, level = evaluation_level(audio[start:end])
    return adjusted, {'version': 'boundary-safe-level-v2', 'start_seconds': start/rate,
                      'end_seconds': end/rate, 'excluded_edge_transients': removed, **level}


def recognize(path, folder, language=None, progress=lambda message: None, fallback_language=None):
    started = active_time()
    from faster_whisper import WhisperModel
    from studio.paths import MODELS
    root = download_asr(progress, language)
    progress('Loading the speech recognizer on CPU')
    load_started = active_time()
    model = WhisperModel(root, device='cpu', compute_type='int8', cpu_threads=2, num_workers=1,
                         download_root=str(MODELS), local_files_only=True)
    loaded = active_time()
    audio_path = decode(path, Path(folder)/'recognition-input.wav', rate=16000, duration=60.1)
    audio, _ = sf.read(audio_path, dtype='float32')
    audio, level = evaluation_level(audio)
    progress('Recognizing the spoken words on CPU')
    recognition_started = active_time()
    # No expected text is passed as an initial prompt: doing so would bias the
    # recognizer toward the answer and invalidate a pronunciation/content test.
    segments, info = model.transcribe(audio, language=language or None, beam_size=5,
        temperature=0.0, vad_filter=True, word_timestamps=True,
        condition_on_previous_text=False, hallucination_silence_threshold=1.0)
    if language is None:
        progress(f'Detected speech language: {info.language} ({info.language_probability:.0%})')
    if language is None and (info.language == 'ml' or
                             fallback_language == 'ml' and info.language not in ('en', 'hi')):
        detection = {'language': info.language, 'probability': info.language_probability}
        del segments, model
        gc.collect()
        progress('Using the native Malayalam recognizer to verify the detected language')
        result = recognize(path, folder, 'ml', progress)
        result['initial_language_detection'] = detection
        return result
    collected = []
    for segment in segments:
        collected.append({'start': segment.start, 'end': segment.end, 'text': segment.text.strip(),
                          'average_log_probability': segment.avg_logprob,
                          'words': [{'start': word.start, 'end': word.end, 'word': word.word,
                                     'probability': word.probability} for word in segment.words or []]})
        progress(f'Recognized through {segment.end:.1f} seconds')
    result = {'text': ' '.join(segment['text'] for segment in collected), 'segments': collected,
              'language': info.language, 'language_probability': info.language_probability,
              'language_was_selected': language is not None,
              'model': MALAYALAM_ASR_ID if language == 'ml' else ASR_ID,
              'revision': MALAYALAM_ASR_REVISION if language == 'ml' else ASR_REVISION,
              'decoding': {'beam_size': 5, 'temperature': 0.0, 'vad_filter': True,
                           'condition_on_previous_text': False, 'hallucination_silence_threshold': 1.0},
              'input_preprocessing': level,
              'timings': {'model_load_seconds': loaded-load_started,
                          'audio_preparation_seconds': recognition_started-loaded,
                          'recognition_seconds': active_time()-recognition_started}}
    del model
    gc.collect()
    result['script_check'] = transcript_script_check(result['text'], language or info.language)
    if language == 'ml':
        from studio.compact_recognition import checkpoint_metadata
        checkpoint = checkpoint_metadata(root)
        if checkpoint:
            result['checkpoint'] = checkpoint
    result['timings']['total_seconds'] = active_time()-started
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
    input_details = {}
    for name, path in [('reference', reference), ('output', candidate)]:
        progress(f'Checking independent speaker identity: {name}')
        decoded = decode(path, Path(folder)/f'verifier-{name}.wav', rate=16000, duration=60)
        audio, _ = sf.read(decoded, dtype='float32')
        audio, input_details[name] = verifier_input(audio)
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
            'input_preprocessing_details': input_details,
            'interpretation': 'Higher means more similar to this verifier. It is not a percentage of identity, '
                              'and no universal indistinguishability threshold is claimed.'}


def full_audit(reference, output, expected, folder, language, progress):
    if not language and expected.strip():
        from studio.languages import resolve_language
        # Script supplies a language hint, never expected words to the decoder.
        inferred = resolve_language(expected, 'auto')
        if inferred in ('ml', 'hi'):
            language = inferred
    verifier = speaker_similarity(reference, output, folder, progress)
    # Speaker sessions are released on function exit before the ASR model loads.
    gc.collect()
    transcript = recognize(output, folder, language, progress)
    errors = content_comparison(expected, transcript)
    if errors.get('available'):
        words = [word for segment in transcript['segments'] for word in segment['words']]
        for difference in errors['differences']:
            index = difference['recognized_word_index']
            if words:
                difference['output_seconds'] = words[min(index, len(words)-1)]['start']
    result = {'speaker_verification': verifier, 'transcription': transcript, 'content_comparison': errors}
    Path(folder, 'quality-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result
