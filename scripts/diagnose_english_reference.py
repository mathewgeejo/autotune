"""Privately inspect saved English reference/output jobs without resynthesis."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio import recognition
from studio.models import download_asr
from studio.runtime import keep_awake
import faster_whisper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    first = args.jobs[0]
    report = {'status': 'running', 'scope': 'Private local reference/content diagnosis. No perceptual accuracy percentage.', 'outputs': []}
    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    def progress(message):
        print(message, flush=True)
    save()
    original = faster_whisper.WhisperModel
    model = None
    try:
        with keep_awake():
            root = download_asr(progress, 'en')
            model = original(root, device='cpu', compute_type='int8', cpu_threads=2, num_workers=1, local_files_only=True)
            # One model for this serial diagnosis; the app still releases its
            # recognizer between phases. No timing comparison is claimed here.
            faster_whisper.WhisperModel = lambda *args, **kwargs: model
            for label, audio in [('reference_full', first/'source.wav'), ('reference_selected', first/'voice-sample.wav')]:
                destination = folder/label
                destination.mkdir()
                report[label] = recognition.recognize(audio, destination, 'en', progress)
                save()
            for job in args.jobs:
                request = json.loads((job/'request.json').read_text(encoding='utf-8'))
                result = json.loads((job/'result.json').read_text(encoding='utf-8'))
                destination = folder/job.name
                destination.mkdir()
                transcript = recognition.recognize(result['audio'], destination, 'en', progress)
                report['outputs'].append({'job': str(job), 'expected': request['text'], 'audio': result['audio'],
                    'transcription': transcript, 'content_comparison': recognition.content_comparison(request['text'], transcript)})
                save()
            faster_whisper.WhisperModel = original
            model = None
            gc.collect()
            for item in report['outputs']:
                destination = folder/Path(item['job']).name
                item['speaker_verification'] = recognition.speaker_similarity(first/'voice-sample.wav',
                    item['audio'], destination, progress)
                save()
            report['status'] = 'measured'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        faster_whisper.WhisperModel = original
        model = None
        save()


if __name__ == '__main__':
    main()
