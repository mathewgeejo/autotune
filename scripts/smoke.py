"""Exercise actual local models, recording elapsed time and peak worker memory.

Usage: .venv/Scripts/python.exe scripts/smoke.py instrument|singer|prepare|link
Singing smoke uses short upstream examples, downloaded only for this check.
"""
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, DATA, REFERENCES
configure()
from studio.jobs import JobManager


def main():
    import numpy as np
    import soundfile as sf
    from studio.audio import decode
    mode = sys.argv[1] if len(sys.argv) > 1 else 'instrument'
    folder = DATA / 'smoke'
    folder.mkdir(exist_ok=True)
    if mode == 'instrument':
        rate = 22050
        # Stable notes with brief rests: tests actual note inference and soundfont rendering.
        chunks = []
        for frequency in (220, 261.6256, 329.6276, 440):
            t = np.arange(rate)/rate
            envelope = np.minimum(1, t*20)*np.minimum(1, (1-t)*20)
            chunks.extend([0.25*np.sin(2*np.pi*frequency*t)*envelope, np.zeros(rate//5)])
        source = folder / 'melody.wav'
        sf.write(source, np.concatenate(chunks), rate)
        payload = {'source': str(source), 'instrument': 'Piano', 'start': 0, 'end': 0}
    else:
        source = folder / 'upstream-singing.wav'
        if not source.exists():
            urllib.request.urlretrieve('https://raw.githubusercontent.com/Plachtaa/seed-vc/51383efd921027683c89e5348211d93ff12ac2a8/examples/source/TECHNOPOLIS%20-%202085%20%5Bvocals%5D_%5Bcut_14sec%5D.wav', source)
        if mode in ('prepare', 'link'):
            payload = ({'file': str(source)} if mode == 'prepare' else {'url':
                'https://raw.githubusercontent.com/Plachtaa/seed-vc/51383efd921027683c89e5348211d93ff12ac2a8/examples/source/TECHNOPOLIS%20-%202085%20%5Bvocals%5D_%5Bcut_14sec%5D.wav'})
        else:
            ref = folder / 'upstream-reference.wav'
            if not ref.exists():
                urllib.request.urlretrieve('https://raw.githubusercontent.com/Plachtaa/seed-vc/51383efd921027683c89e5348211d93ff12ac2a8/examples/reference/s1p1.wav', ref)
            cache = REFERENCES / 'smoke'
            cache.mkdir(exist_ok=True)
            target = decode(ref, cache/'vocals.wav', duration=5)
            payload = {'source': str(source), 'reference': str(target), 'start': 0, 'end': 5,
                       'ref_start': 0, 'ref_duration': 5, 'steps': 10}
    manager = JobManager()
    job_id = manager.submit('prepare' if mode == 'link' else mode, payload)
    previous = ''
    while True:
        job = manager.snapshot(job_id)
        stage = job['status']['stage']
        if stage != previous:
            print(stage, flush=True)
            previous = stage
        if job['state'] != 'running':
            print(json.dumps({'state': job['state'], 'seconds': round(job['elapsed'], 1),
                              'peak_mb': round(job['peak_mb']), 'folder': str(job['folder']),
                              'result': job.get('result')}, indent=2), flush=True)
            if job['state'] != 'complete':
                print((job['folder']/'worker.log').read_text(encoding='utf-8', errors='replace')[-8000:])
                sys.exit(1)
            if mode == 'instrument':
                import pretty_midi
                midi = pretty_midi.PrettyMIDI(job['result']['midi'])
                pitches = {n.pitch for i in midi.instruments for n in i.notes}
                assert {57, 60, 64, 69}.issubset(pitches), f'Missing expected notes: {pitches}'
                assert sf.info(job['result']['audio']).duration > 4
            return
        time.sleep(2)


if __name__ == '__main__':
    main()
