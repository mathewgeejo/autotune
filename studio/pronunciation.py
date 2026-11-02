"""Optional Malayalam wording check with one voice-guarded retry.

Large models are loaded sequentially. The caller releases its synthesizer before
entering this function; each recognizer/verifier releases its model on return.
"""
import gc

from studio.accelerators import WarmShapeFlow
from studio.runtime import active_time
from studio.voice_selection import compare_reading

CHECKED_QUALITY = 'Detailed + word check'
INDIC_QUALITIES = ('Balanced', 'Detailed', CHECKED_QUALITY)


def check_pronunciation(request, folder, prepared, default_result, progress, write_report):
    from studio.recognition import recognize, content_comparison, speaker_similarity, full_audit
    from studio.indic import IndicEngine
    result = dict(default_result)
    started = active_time()
    report = {'status': 'checking', 'selected': 'default', 'default_audio': default_result['audio'],
              'expected_words_supplied_to_recognizer': False,
              'scope': 'Automatic wording/voice comparison; no human indistinguishability claim.',
              'default_precision': default_result.get('precision'), 'retry_attempted': False}
    checks = folder/'word-check'
    try:
        checks.mkdir(exist_ok=True)
        progress('Checking the generated Malayalam wording')
        transcript = recognize(default_result['audio'], checks, 'ml', progress)
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
                progress('Trying a more text-guided Malayalam reading')
                engine = None
                try:
                    engine = IndicEngine(progress, 'Detailed')
                    engine.guidance_scale = 3.0
                    engine.precision = {**engine.precision, 'guidance_scale': 3.0}
                    if getattr(engine.flow_decoder, 'backend', '').startswith('openvino'):
                        engine.flow_decoder = WarmShapeFlow(engine.flow_decoder)
                        engine.precision['gpu_shape_warmup_evaluations'] = 3
                    candidate = engine.synthesize({**request, 'quality': 'Detailed'}, retry_folder,
                                                  progress, prepared=prepared)
                    if isinstance(engine.flow_decoder, WarmShapeFlow):
                        report['retry_gpu_warmup'] = {
                            'extra_evaluations': engine.flow_decoder.extra_evaluations,
                            'elapsed_seconds': engine.flow_decoder.warmup_seconds,
                            'shapes': sorted(engine.flow_decoder.shapes)}
                finally:
                    engine = None
                    gc.collect()
                report.update(candidate_audio=candidate['audio'], candidate_precision=candidate.get('precision'))
                progress('Comparing the new reading with the original voice and wording')
                candidate_audit = full_audit(default_result['reference_preview'], candidate['audio'],
                                             request['text'], retry_folder, 'ml', progress)
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
