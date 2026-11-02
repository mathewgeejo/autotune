"""Measure generated speech independently; retain bad results instead of hiding them."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.analysis import analyze_files
from studio.recognition import full_audit
from studio.runtime import keep_awake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'data/smoke/quality-benchmark.json')
    args = parser.parse_args()
    results = []
    with keep_awake():
        for report in args.reports:
            entries = json.loads(report.read_text(encoding='utf-8'))
            for entry in entries:
                result = entry['result']
                folder = ROOT/'data/smoke/quality'/entry['job_id']
                folder.mkdir(parents=True, exist_ok=True)
                def progress(message):
                    print(f"{entry['job_id'][:8]} | {message}", flush=True)
                try:
                    acoustics, _, _, _ = analyze_files(result['reference_preview'], result['audio'], folder)
                    audit = full_audit(result['reference_preview'], result['audio'], entry['expected'], folder,
                                       entry['language'], progress)
                    record = {'job_id': entry['job_id'], 'language': entry['language'],
                              'quality': entry['quality'], 'expected': entry['expected'],
                              'source_report': str(report), 'acoustics': acoustics, **audit}
                    # These are explicit engineering review targets, not a
                    # claim that passing them establishes indistinguishability.
                    content = audit['content_comparison']
                    speaker = audit['speaker_verification']
                    record['needs_review'] = (not content.get('available') or content.get('word_error_rate', 1) > 0.15
                        or not speaker.get('available') or speaker.get('cosine_similarity', -1) < 0.8)
                    results.append(record)
                    print(json.dumps({'job': entry['job_id'], 'recognized': audit['transcription']['text'],
                        'speaker_cosine': speaker.get('cosine_similarity'),
                        'word_error_rate': content.get('word_error_rate'),
                        'needs_review': record['needs_review']}, ensure_ascii=False), flush=True)
                except Exception as exc:
                    results.append({'job_id': entry['job_id'], 'language': entry['language'],
                                    'verification_failed': True, 'error': str(exc)})
                    print(f'Verification failed: {exc}', flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
