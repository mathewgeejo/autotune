"""Measure an isolated recognizer checkpoint against saved real-audio controls.

All audio preparation and decoding use the production recognizer function. The
only override is the local checkpoint directory; no expected-text prompt is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio import recognition
from studio.analysis import normalize_text
from studio.runtime import keep_awake
import psutil


class PeakMemory:
    def __enter__(self):
        self.done = threading.Event()
        self.resident, self.private = 0, 0
        process = psutil.Process()
        def sample():
            while True:
                info = process.memory_info()
                self.resident = max(self.resident, info.rss)
                self.private = max(self.private, getattr(info, 'private', 0))
                if self.done.wait(.1):
                    return
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--baseline-report', required=True, nargs='+', type=Path)
    parser.add_argument('--include-natural-controls', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    root = args.model_root.resolve(strict=True)
    model = root/'model.bin'
    digest = hashlib.sha256()
    with model.open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            digest.update(block)
    manifest_path = root/'conversion.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    if manifest and (manifest['state'] != 'generated-unvalidated' or manifest['model_sha256'] != digest.hexdigest()):
        parser.error('The generated checkpoint manifest or hash is invalid.')
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'model_root': str(root), 'model_sha256': digest.hexdigest(),
              'model_bytes': model.stat().st_size, 'conversion': manifest,
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Recognizer storage/decoding comparison; no synthesis or listening quality claim.',
              'cases': []}
    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    cases = []
    for path in args.baseline_report:
        baseline = json.loads(path.read_text(encoding='utf-8'))
        if baseline['language'] != 'ml':
            parser.error('Use Malayalam baselines with the same pinned native recognizer.')
        cases.append((path.stem, baseline['result']['audio'], baseline['expected'], baseline['candidate_audit']))
        if args.include_natural_controls and 'natural_content_control' in baseline:
            natural = baseline['natural_content_control']
            cases.append((path.stem+'-natural', natural['audio'], baseline['expected'], natural))
    original_download = recognition.download_asr
    def local_checkpoint(progress, language):
        if language != 'ml':
            raise ValueError('This trial is limited to the native Malayalam recognizer')
        return str(root)
    recognition.download_asr = local_checkpoint
    progress = lambda message: print(message, flush=True)
    save()
    try:
        with keep_awake():
            for index, (name, audio, expected, baseline) in enumerate(cases):
                destination = folder/f'{index+1}-{name}'
                destination.mkdir()
                case = {'name': name, 'audio': str(audio), 'state': 'running'}
                report['cases'].append(case)
                save()
                before = baseline['transcription']
                with PeakMemory() as memory:
                    actual = recognition.recognize(audio, destination, 'ml', progress)
                if (actual['model'], actual['revision']) != (before['model'], before['revision']):
                    raise ValueError('The source recognizer model or revision is incomparable')
                case.update(state='measured', transcription=actual,
                    baseline_transcription=before,
                    exact_transcript_match=actual['text'] == before['text'],
                    normalized_transcript_match=normalize_text(actual['text']) == normalize_text(before['text']),
                    content_comparison=recognition.content_comparison(expected, actual),
                    baseline_content_comparison=baseline['content_comparison'],
                    sampled_peak_resident_mb=memory.resident/2**20,
                    sampled_peak_private_mb=memory.private/2**20 if memory.private else None)
                print(json.dumps({'name': name, 'exact_match': case['exact_transcript_match'],
                                  'timings': actual['timings']}, ensure_ascii=False), flush=True)
                save()
        report['status'] = 'measured'
        report['all_exact_transcript_matches'] = all(case['exact_transcript_match'] for case in report['cases'])
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        recognition.download_asr = original_download
        save()


if __name__ == '__main__':
    main()
