"""Generate over one minute of real speech and verify every passage is retained."""
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


TEXT = '''Good morning. Today we are taking a walk through the old part of town.
The first street is quiet, and the little bakery has just opened its doors.
I can smell fresh bread as we pass the window. A woman waves from the counter.
We turn left at the corner and follow the path beside the river.
The water moves slowly under the bridge, and a few birds rest on the railing.
Across the road there is a small garden with red flowers and a wooden bench.
Let us stop here for a moment and look at the map before we continue.
Our next stop is the library, where the afternoon reading begins at two.
Please remember to bring the blue notebook and the letter from your friend.
After the reading we can meet outside and decide where to have some tea.
There is no need to hurry. We have enough time to enjoy the journey together.
On the way back we will visit the market and choose something for dinner.
I would like a few apples, some rice, and the vegetables we saw yesterday.
If it starts to rain, we can wait under the covered entrance near the station.
The evening train will arrive soon, and our friends will be happy to see us.
Thank you for joining me on this walk and for sharing such a pleasant day.
The final words of this recording are welcome home and see you tomorrow.'''


def main():
    manager = JobManager()
    try:
        job_id = manager.submit('clone', {'source': str(ROOT/'data/smoke/dave.wav'),
                                        'text': TEXT, 'language': 'en'})
        job = wait(manager, job_id, timeout=2400)
        result = job['result']
        samples, rate = sf.read(result['audio'], dtype='float32')
        assert np.isfinite(samples).all() and np.max(np.abs(samples)) <= 0.981
        assert result['duration'] >= 60, result['duration']
        parts = [job['folder']/f'passage-{i+1}.wav' for i in range(result['passages'])]
        assert len(samples) == sum(sf.info(part).frames for part in parts)
        position = 0
        for part in parts:
            audio, _ = sf.read(part, dtype='float32')
            np.testing.assert_array_equal(samples[position:position+len(audio)], audio)
            position += len(audio)
        report = {'job_id': job_id, 'duration_seconds': len(samples)/rate,
                  'passages': result['passages'], 'frames': len(samples),
                  'passages_preserved_sample_for_sample': True,
                  'expected_text': TEXT, 'result': result,
                  'elapsed_seconds': job['elapsed'], 'peak_memory_mb': job['peak_mb'],
                  'scope': 'Generation and complete passage assembly; no claim of full-text pronunciation accuracy.'}
        target = ROOT/'data/smoke/minute-output.json'
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        manager.close()


if __name__ == '__main__':
    main()
