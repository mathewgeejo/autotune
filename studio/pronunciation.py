"""Optional English/Malayalam wording check with one voice-guarded retry.

Large models are loaded sequentially. The caller releases its synthesizer before
entering this function; each recognizer/verifier releases its model on return.
"""
import gc

from studio.accelerators import WarmShapeFlow
from studio.runtime import active_time
from studio.voice_selection import compare_reading

CHECKED_QUALITY = 'Detailed + word check'
INDIC_QUALITIES = ('Balanced', 'Detailed', CHECKED_QUALITY)


def reading_transcript(result, folder, language, progress):
    from pathlib import Path
    import soundfile as sf
    from studio.recognition import recognize
    if result.get('duration', 0) <= 60:
        return recognize(result['audio'], folder, language, progress)
    # Recognition accepts at most one minute. Check every closed passage for
    # longer exports instead of silently scoring only the beginning of a WAV.
    count = result.get('passages', 0)
    if not count:
        raise ValueError('The complete passage list is required to check a long export.')
    transcripts, offset = [], 0
    for index in range(1, count+1):
        part = Path(result['audio']).parent/f'passage-{index}.wav'
        duration = sf.info(part).duration
        if duration > 60:
            raise ValueError('An individual passage exceeds the complete recognition limit.')
        destination = folder/f'passage-{index}'
        destination.mkdir(parents=True, exist_ok=True)
        transcript = recognize(part, destination, language, progress)
        if transcripts and any(transcript.get(key) != transcripts[0].get(key) for key in ('model', 'revision')):
            raise ValueError('Recognition models changed while checking the export.')
        if not transcript.get('script_check', {}).get('reliable'):
            return transcript
        for segment in transcript.get('segments', []):
            segment['start'] += offset
            segment['end'] += offset
            for word in segment.get('words', []):
                word['start'] += offset
                word['end'] += offset
        transcripts.append(transcript)
        offset += duration
    return {**transcripts[0], 'text': ' '.join(item['text'] for item in transcripts),
            'segments': [segment for item in transcripts for segment in item.get('segments', [])],
            'checked_passages': count}


def check_pronunciation(request, folder, prepared, default_result, progress, write_report):
    from studio.recognition import content_comparison, speaker_similarity, full_audit
    from studio.indic import IndicEngine
    result = dict(default_result)
    language = request.get('language', 'ml')
    english = language == 'en'
    language_name = 'English' if english else 'Malayalam'
    started = active_time()
    report = {'status': 'checking', 'selected': 'default', 'default_audio': default_result['audio'],
              'expected_words_supplied_to_recognizer': False,
              'scope': 'Automatic wording/voice comparison; no human indistinguishability claim.',
              'default_precision': default_result.get('precision'), 'retry_attempted': False}
    checks = folder/'word-check'
    try:
        checks.mkdir(exist_ok=True)
        progress(f'Checking the generated {language_name} wording')
        transcript = reading_transcript(default_result, checks, language, progress)
        content = content_comparison(request['text'], transcript)
        report['default_audit'] = {'transcription': transcript, 'content_comparison': content}
        if not content.get('available'):
            report.update(status='check-unavailable', reason=content.get('reason'))
        elif content['word_error_rate'] == 0 and content['grapheme_error_rate'] == 0:
            report.update(status='word-check-passed', reason='No recognized word/grapheme differences; no retry needed.')
        else:
            progress('Checking voice similarity before attempting another reading')
            speaker = speaker_similarity(default_result['reference_preview'], default_result['audio'], checks, progress)
            report['default_audit']['speaker_verification'] = speaker
            if not speaker.get('available'):
                report.update(status='default-retained', reason='Voice comparison is unavailable, so the original reading is retained.')
            else:
                retry_folder = folder/'pronunciation-retry'
                retry_folder.mkdir()
                report['retry_attempted'] = True
                progress(f'Trying a more text-guided {language_name} reading')
                engine = None
                try:
                    if english:
                        from studio.speech import SpeechEngine
                        engine = SpeechEngine(progress)
                        engine.english_max_words = 8
                        engine.english_max_chars = 80
                    else:
                        engine = IndicEngine(progress, 'Detailed')
                        engine.guidance_scale = 3.0
                        engine.precision = {**engine.precision, 'guidance_scale': 3.0}
                    if not english and getattr(engine.flow_decoder, 'backend', '').startswith('openvino'):
                        engine.flow_decoder = WarmShapeFlow(engine.flow_decoder)
                        engine.precision['gpu_shape_warmup_evaluations'] = 3
                    candidate = engine.synthesize({**request, 'quality': 'Detailed'}, retry_folder,
                                                  progress, prepared=prepared)
                    if not english and isinstance(engine.flow_decoder, WarmShapeFlow):
                        report['retry_gpu_warmup'] = {
                            'extra_evaluations': engine.flow_decoder.extra_evaluations,
                            'elapsed_seconds': engine.flow_decoder.warmup_seconds,
                            'shapes': sorted(engine.flow_decoder.shapes)}
                finally:
                    engine = None
                    gc.collect()
                report.update(candidate_audio=candidate['audio'], candidate_precision=candidate.get('precision'))
                progress('Comparing the new reading with the original voice and wording')
                if candidate.get('duration', 0) <= 60:
                    candidate_audit = full_audit(default_result['reference_preview'], candidate['audio'],
                                                 request['text'], retry_folder, language, progress)
                else:
                    heard = reading_transcript(candidate, retry_folder, language, progress)
                    candidate_audit = {'transcription': heard,
                        'content_comparison': content_comparison(request['text'], heard),
                        'speaker_verification': speaker_similarity(default_result['reference_preview'],
                                                                  candidate['audio'], retry_folder, progress)}
                report['candidate_audit'] = candidate_audit
                decision = compare_reading(report['default_audit'], candidate_audit)
                report.update(status='compared', decision=decision)
                if decision['accepted']:
                    result = dict(candidate)
                    report['selected'] = 'retry'
                    progress('The new reading improves the recognized wording while retaining voice similarity')
                else:
                    progress('Keeping the original reading after the wording and voice comparison')
    except Exception as exc:
        # A valid completed synthesis must survive an optional download, checker,
        # or retry failure. All audio and any completed checks remain available.
        report.update(status='check-unavailable', error=f'{type(exc).__name__}: {exc}')
        progress('The automatic wording check could not finish; keeping the generated speech')
    report['elapsed_seconds'] = active_time()-started
    report['selected_audio'] = result['audio']
    report['render_synthesis_seconds'] = default_result['synthesis_seconds']
    if 'candidate_audio' in report:
        report['render_synthesis_seconds'] += candidate['synthesis_seconds']
    result['word_check'] = report
    try:
        write_report(report)
    except OSError:
        progress('The separate word-check report could not be saved; details remain in the result')
    return result
