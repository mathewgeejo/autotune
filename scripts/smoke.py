"""Run real sample-to-text speech cloning and measure cold/warm generation.

python scripts/smoke.py --reference data/smoke/dave.wav --repeat 2
Samples are never fabricated; provide a real speech recording.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.jobs import JobManager
from studio.runtime import active_time
import numpy as np
import soundfile as sf


def wait(manager, job_id, timeout=1800):
    deadline = active_time() + timeout
    previous = None
    while active_time() < deadline:
        job = manager.snapshot(job_id)
        message = job['status']['stage']
        if message != previous:
            print(f"{job_id[:8]} | {int(job['elapsed'])}s | {message}", flush=True)
            previous = message
        if job['state'] != 'running':
            if job['state'] != 'complete':
                raise RuntimeError(f"{job['state']}: {message}; log: {job['folder'] / 'worker.log'}")
            return job
        time.sleep(0.5)
    manager.cancel()
    raise TimeoutError('Speech generation did not finish within the smoke-test deadline.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--text', default='Hello there. This is a new sentence, spoken using the voice in the sample.')
    parser.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    manager = JobManager()
    results = []
    try:
        worker_pid = None
        for index in range(args.repeat):
            job_id = manager.submit('clone', {'source': str(args.reference.resolve()), 'text': args.text})
            job = wait(manager, job_id)
            pid = job['process'].pid
            if worker_pid is not None:
                assert pid == worker_pid, 'Warm generation restarted the model worker'
                assert job['result']['cached_voice'], 'Warm generation did not reuse conditioning'
            worker_pid = pid
            audio, rate = sf.read(job['result']['audio'])
            assert rate == 24000 and len(audio) > rate
            assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.01
            result = {**job['result'], 'elapsed_seconds': job['elapsed'],
                      'peak_memory_mb': job['peak_mb'], 'job_id': job_id, 'worker_pid': pid}
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
        target = ROOT / 'data/smoke/cloning-report.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'Real speech cloning passed. Report: {target}', flush=True)
    finally:
        manager.close()


if __name__ == '__main__':
    main()
