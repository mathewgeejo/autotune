"""Real offline native cloning with blank reference words and automatic recognition."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
from studio.paths import configure
configure()
from studio.jobs import JobManager
from studio.recognition import full_audit, transcript_script_check
from studio.runtime import keep_awake
from scripts.multilingual_smoke import SENTENCES
from scripts.smoke import wait
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['ml', 'hi'], default='ml')
    args = parser.parse_args()
    language = args.language
    manager = JobManager()
    with keep_awake():
        try:
            job_id = manager.submit('clone', {'source': str(ROOT/f'data/smoke/fleurs-{language}_in/reference.wav'),
                'text': SENTENCES[language], 'language': 'auto', 'quality': 'Detailed',
                'reference_text': '', 'reference_language': 'auto'})
            job = wait(manager, job_id, timeout=1800)
            result = job['result']
            assert result['language'] == language
            assert transcript_script_check(result['reference_text'], language)['reliable']
            audio, rate = sf.read(result['audio'], dtype='float32')
            assert rate == 24000 and np.isfinite(audio).all() and len(audio) > 24000
            report = {'job_id': job_id, 'result': result, 'elapsed_seconds': job['elapsed'],
                      'peak_memory_mb': job['peak_mb'], 'offline': True,
                      'reference_words_supplied': False, 'reference_language_supplied': False}
        finally:
            manager.close()
        gc.collect()
        report['audit'] = full_audit(result['reference_preview'], result['audio'], SENTENCES[language],
                                    job['folder'], language, lambda message: print(message, flush=True))
        (ROOT/f'data/smoke/automatic-indic-{language}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
