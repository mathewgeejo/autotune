"""Calibrate the existing generic Whisper checkpoint on forced-language Malayalam.

Natural same-words audio is transcribed first. Expected words are used only by
the scorer after decoding, never as a recognizer prompt. Production ASR routing
is unchanged by this diagnostic.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.audio import decode
from studio.models import download_asr, ASR_ID, ASR_REVISION
from studio.recognition import evaluation_level, transcript_script_check, content_comparison
from studio.runtime import active_time, keep_awake
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--candidate-reports', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path to preserve prior calibration.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    if baseline['language'] != 'ml':
        parser.error('This diagnostic requires a known Malayalam case.')
    cases = [('natural-control', baseline['natural_content_control']['audio']),
             ('baseline-synthesis', baseline['result']['audio'])]
    for path in args.candidate_reports:
        item = json.loads(path.read_text(encoding='utf-8'))
        if item['expected'].strip() != baseline['expected'].strip():
            raise ValueError('Candidate expected words differ from the natural control')
        cases.append((path.stem, item['result']['audio']))
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    report = {'model': ASR_ID, 'revision': ASR_REVISION, 'forced_language': 'ml',
              'expected_text_supplied_to_recognizer': False, 'baseline_report': str(args.baseline_report),
              'status': 'running', 'cases': [],
              'scope': 'Cross-check between Whisper checkpoints; no human pronunciation verdict or production routing change.'}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(args.output)
    progress = lambda message: print(message, flush=True)
    save()
    with keep_awake():
        from faster_whisper import WhisperModel
        model = WhisperModel(download_asr(progress), device='cpu', compute_type='int8', cpu_threads=2,
                             num_workers=1, local_files_only=True)
        for name, path in cases:
            destination = folder/name
            destination.mkdir()
            audio_path = decode(path, destination/'recognition-input.wav', rate=16000, duration=60.1)
            audio, _ = sf.read(audio_path, dtype='float32')
            audio, level = evaluation_level(audio)
            started = active_time()
            segments, info = model.transcribe(audio, language='ml', beam_size=5, temperature=0.0,
                vad_filter=True, word_timestamps=True, condition_on_previous_text=False,
                hallucination_silence_threshold=1.0)
            collected = []
            for segment in segments:
                collected.append({'start': segment.start, 'end': segment.end, 'text': segment.text.strip(),
                                  'average_log_probability': segment.avg_logprob})
            transcript = {'text': ' '.join(item['text'] for item in collected), 'segments': collected,
                          'language': info.language, 'language_was_selected': True,
                          'model': ASR_ID, 'revision': ASR_REVISION, 'input_preprocessing': level}
            transcript['script_check'] = transcript_script_check(transcript['text'], 'ml')
            record = {'case': name, 'audio': str(path), 'seconds': active_time()-started,
                      'transcription': transcript,
                      'content_comparison': content_comparison(baseline['expected'], transcript)}
            report['cases'].append(record)
            save()
            print(json.dumps(record, ensure_ascii=False), flush=True)
        report['status'] = 'measured'
        control = report['cases'][0]['content_comparison']
        rejected = not control.get('available') or control.get('grapheme_error_rate', 1) > 0.5
        report['natural_control_calibration'] = {
            'rejected_as_pronunciation_judge': rejected,
            'reason': ('Natural control fails script validation or has more than half its graphemes misrecognized.'
                       if rejected else 'Not rejected by this control; this single case does not establish general recognizer accuracy.')}
        save()


if __name__ == '__main__':
    main()
