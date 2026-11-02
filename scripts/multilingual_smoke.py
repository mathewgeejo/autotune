"""Actual native-script synthesis; never substitutes an English-only engine."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.jobs import JobManager
from scripts.smoke import wait
import numpy as np
import soundfile as sf

SENTENCES = {'hi': 'नमस्ते। आज मौसम अच्छा है।',
             'ml': 'നമസ്കാരം. ഇന്ന് നല്ല ദിവസമാണ്.',
             'en': 'Hello. Today is a wonderful day.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=ROOT/'data/smoke/dave.wav')
    parser.add_argument('--reference-text', type=Path, default=ROOT/'data/smoke/dave.txt')
    parser.add_argument('--quality', choices=['Balanced', 'Detailed'], default='Detailed')
    parser.add_argument('--languages', nargs='+', default=['hi', 'ml', 'en'])
    args = parser.parse_args()
    manager = JobManager()
    reports = []
    try:
        for language in args.languages:
            request = {'source': str(args.reference.resolve()), 'text': SENTENCES[language],
                       'language': language, 'quality': args.quality,
                       'reference_text': args.reference_text.read_text(encoding='utf-8')}
            job_id = manager.submit('clone', request)
            job = wait(manager, job_id, timeout=1800)
            result = job['result']
            audio, rate = sf.read(result['audio'])
            assert rate == 24000 and len(audio) > rate*0.3
            assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.01
            assert result['language'] == language
            if language in ('ml', 'hi'):
                assert 'DhVaani' in result['model']
                assert result.get('reference_text')
            reports.append({'job_id': job_id, 'language': language, 'quality': args.quality,
                            'expected': request['text'], 'result': result,
                            'elapsed_seconds': job['elapsed'], 'peak_memory_mb': job['peak_mb']})
            print(json.dumps(reports[-1], ensure_ascii=False, indent=2), flush=True)
            target = ROOT/'data/smoke'/f'multilingual-{args.quality.lower()}.json'
            target.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        manager.close()


if __name__ == '__main__':
    main()
