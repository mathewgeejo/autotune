"""Measure whether fewer FP32 flow steps retain the fixed baseline's voice and words."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.indic import IndicEngine
from studio.recognition import full_audit
from studio.runtime import keep_awake
from studio.speech import prepare_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', choices=[8, 16, 24, 32], default=[8, 16])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    root = args.output.parent/args.output.stem
    root.mkdir(parents=True, exist_ok=True)
    reference_folder = root/'reference'
    reference_folder.mkdir(exist_ok=True)
    prepared = prepare_sample(baseline['source_reference'], reference_folder)
    old_preview = Path(baseline['result']['reference_preview'])
    assert hashlib.sha256(old_preview.read_bytes()).digest() == hashlib.sha256(prepared[1].read_bytes()).digest(), 'The selected reference changed; this is not a controlled comparison.'
    report = {'baseline_report': str(args.baseline_report), 'comparison': 'Fixed reference, text, FP32 weights, guidance 2 and seed 666; only flow steps vary.',
              'scope': 'Automated engineering comparison; perceptual equivalence remains unproven.', 'trials': []}
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    progress = lambda message, **kwargs: print(message, flush=True)
    with keep_awake():
        engine = IndicEngine(progress, 'Detailed')
        for steps in args.steps:
            folder = root/f'steps-{steps}'
            folder.mkdir(exist_ok=True)
            if (folder/'speech.wav').exists():
                raise RuntimeError('Use a fresh output name to preserve prior trials.')
            engine.steps = steps
            engine.precision = {**engine.precision, 'steps': steps}
            print(f'TRIAL {steps} steps', flush=True)
            result = engine.synthesize({'source': baseline['source_reference'], 'text': baseline['expected'],
                'language': baseline['language'], 'reference_text': baseline['result']['reference_text']},
                folder, progress, prepared)
            report['trials'].append({'steps': steps, 'result': result, 'state': 'generated'})
            save()
        del engine
        gc.collect()
        for trial in report['trials']:
            result = trial['result']
            folder = root/f"steps-{trial['steps']}"
            trial['audit'] = full_audit(result['reference_preview'], result['audio'], baseline['expected'],
                                       folder, baseline['language'], progress)
            trial['state'] = 'measured'
            save()
            print(json.dumps({'steps':trial['steps'],'synthesis_seconds':result['synthesis_seconds'],
                              'speaker_cosine':trial['audit']['speaker_verification'].get('cosine_similarity'),
                              'content':trial['audit']['content_comparison']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
