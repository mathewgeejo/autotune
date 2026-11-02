"""Controlled, explicitly experimental Indic guidance/rate trials, not a quality pass."""
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.indic import IndicEngine
from studio.speech import prepare_sample
from studio.runtime import keep_awake, active_time
from studio.recognition import full_audit
from scripts.multilingual_smoke import SENTENCES


def main():
    output = ROOT/'data/smoke/indic-trials'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    trials = [('native-guidance-two', ROOT/'data/smoke/fleurs-ml_in/reference.wav',
               ROOT/'data/smoke/fleurs-ml_in/reference.txt', 1.0, 2.0),
              ('cross-language-slower', ROOT/'data/smoke/dave.wav',
               ROOT/'data/smoke/dave.txt', 0.45, 1.5)]
    with keep_awake():
        for name, reference, caption, speed, guidance in trials:
            folder = output/name
            folder.mkdir(exist_ok=True)
            def progress(message, **kwargs):
                print(name, message, flush=True)
            prepared = prepare_sample(reference, folder)
            metrics = folder/'speech-metrics.json'
            if metrics.is_file():
                result = json.loads(metrics.read_text(encoding='utf-8'))
                elapsed = result['synthesis_seconds']
                progress('Reusing the completed experimental waveform for evaluation')
            else:
                engine = IndicEngine(progress, 'Detailed')
                engine.speed, engine.guidance_scale = speed, guidance
                engine.precision.update(speed=speed, guidance_scale=guidance)
                started = active_time()
                result = engine.synthesize({'source': str(reference), 'text': SENTENCES['ml'], 'language': 'ml',
                                            'reference_text': caption.read_text(encoding='utf-8')}, folder, progress, prepared)
                elapsed = active_time()-started
                del engine
                gc.collect()
            audit = full_audit(result['reference_preview'], result['audio'], SENTENCES['ml'], folder, 'ml', progress)
            record = {'name': name, 'result': result, 'synthesis_seconds': elapsed, **audit}
            records.append(record)
            (output/'trials.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps({'name': name, 'duration': result['duration'], 'speaker': audit['speaker_verification'],
                              'transcription': audit['transcription']['text'], 'content': audit['content_comparison']},
                              ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
