"""Controlled FP32 step, guidance, and timing comparisons against a fixed voice/text."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.indic import IndicEngine
from studio.recognition import full_audit
from studio.runtime import keep_awake, active_time
from studio.speech import prepare_sample


def legacy_text_conditioning(self, text, prompt_tokens, prompt_length):
    """Pre-experiment duration control, retained for a fair audio comparison."""
    feeds = {'tokens': self.text_ids(text), 'prompt_tokens': prompt_tokens,
             'prompt_features_len': np.array(prompt_length, dtype=np.int64),
             'speed': np.array(1.0, dtype=np.float32)}
    condition = self.text_encoder.run(None, feeds)[0].astype(np.float32)
    if self.speed != 1.0:
        total = condition.shape[1]
        desired = prompt_length+(total-prompt_length)/self.speed
        feeds['speed'] = np.array(total/desired, dtype=np.float32)
        condition = self.text_encoder.run(None, feeds)[0].astype(np.float32)
    return condition


def reference_fixed_text_conditioning(self, text, prompt_tokens, prompt_length):
    """Rejected production candidate, retained for reproducible quality trials."""
    feeds = {'tokens': self.text_ids(text), 'prompt_tokens': prompt_tokens,
             'prompt_features_len': np.array(prompt_length, dtype=np.int64),
             'speed': np.array(1.0, dtype=np.float32)}
    condition = self.text_encoder.run(None, feeds)[0].astype(np.float32)
    if self.speed != 1.0 and condition.shape[1] > prompt_length:
        target = condition[:, prompt_length:]
        count = max(1, round(target.shape[1]/self.speed))
        indices = np.linspace(0, target.shape[1]-1, count).round().astype(np.int64)
        condition = np.concatenate((condition[:, :prompt_length], target[:, indices]), axis=1)
    return condition


class WarmShapeFlow:
    """Experimental repeated first-step evaluation; only the last value advances the solver."""
    def __init__(self, flow, evaluations):
        self.flow, self.evaluations, self.shapes = flow, evaluations, set()
        self.warmup_seconds = 0

    @property
    def backend(self):
        return getattr(self.flow, 'backend', 'onnxruntime-cpu')

    def run(self, output_names, feeds):
        shape = tuple(feeds['x'].shape)
        if shape not in self.shapes and self.backend.startswith('openvino'):
            started = active_time()
            for _ in range(self.evaluations-1):
                self.flow.run(output_names, feeds)
                if not self.backend.startswith('openvino'):
                    break
            self.warmup_seconds += active_time()-started
            self.shapes.add(shape)
        return self.flow.run(output_names, feeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', choices=[8, 16, 24, 32, 64], default=[8, 16])
    parser.add_argument('--guidance', type=float, nargs='+', default=[2.0])
    parser.add_argument('--speed', type=float, nargs='+', default=[1.0])
    parser.add_argument('--accelerator', choices=['cpu', 'intel'], default='cpu')
    parser.add_argument('--timing-mode', choices=['production', 'legacy', 'reference-fixed'], default='production',
                        help='Production uses graph duration control; reference-fixed retains the rejected target-frame experiment.')
    parser.add_argument('--gpu-warmup-evaluations', type=int, choices=[0, 3], default=0,
                        help='Experimental cold-shape stabilization; no production backend change.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path to preserve prior measurements.')
    if any(not 0.25 <= speed <= 4 for speed in args.speed):
        parser.error('Speed must be between 0.25 and 4.')
    if any(not 0 <= guidance <= 5 for guidance in args.guidance):
        parser.error('Guidance must be between 0 and 5.')
    os.environ['VOICE_STUDIO_ACCELERATOR'] = args.accelerator
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    root = args.output.parent/args.output.stem
    root.mkdir(parents=True, exist_ok=True)
    reference_folder = root/'reference'
    reference_folder.mkdir(exist_ok=True)
    prepared = prepare_sample(baseline['source_reference'], reference_folder)
    old_preview = Path(baseline['result']['reference_preview'])
    assert hashlib.sha256(old_preview.read_bytes()).digest() == hashlib.sha256(prepared[1].read_bytes()).digest(), 'The selected reference changed; this is not a controlled comparison.'
    report = {'baseline_report': str(args.baseline_report),
              'comparison': 'Fixed reference, text, FP32 weights and seed 666; record steps, guidance, speed and actual backend for each trial.',
              'requested_accelerator': args.accelerator,
              'timing_mode': args.timing_mode,
              'gpu_warmup_evaluations': args.gpu_warmup_evaluations,
              'timing_algorithm': 'target-frame-nearest' if args.timing_mode == 'reference-fixed' else 'graph-target-duration',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'baseline_precision': baseline['result']['precision'],
              'scope': 'Automated engineering comparison; perceptual equivalence remains unproven.', 'trials': []}
    if 'natural_content_control' in baseline:
        report['natural_content_control'] = baseline['natural_content_control']
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    progress = lambda message, **kwargs: print(message, flush=True)
    with keep_awake():
        engine = IndicEngine(progress, 'Detailed')
        if args.gpu_warmup_evaluations:
            engine.flow_decoder = WarmShapeFlow(engine.flow_decoder, args.gpu_warmup_evaluations)
            engine.precision['gpu_shape_warmup_evaluations'] = args.gpu_warmup_evaluations
        if args.timing_mode == 'legacy':
            engine.text_conditioning = MethodType(legacy_text_conditioning, engine)
        elif args.timing_mode == 'reference-fixed':
            engine.text_conditioning = MethodType(reference_fixed_text_conditioning, engine)
        engine.precision['rate_control'] = report['timing_algorithm']
        for steps in args.steps:
            for guidance in args.guidance:
                for speed in args.speed:
                    name = f'steps-{steps}-guidance-{guidance:g}-speed-{speed:g}'
                    folder = root/name
                    folder.mkdir(exist_ok=True)
                    if (folder/'speech.wav').exists():
                        raise RuntimeError('Use a fresh output name to preserve prior trials.')
                    engine.steps, engine.guidance_scale, engine.speed = steps, guidance, speed
                    engine.precision = {**engine.precision, 'steps': steps, 'guidance_scale': guidance, 'speed': speed}
                    print(f'TRIAL {name}', flush=True)
                    trial = {'name': name, 'steps': steps, 'guidance': guidance, 'speed': speed, 'state': 'running'}
                    report['trials'].append(trial)
                    save()
                    try:
                        trial['result'] = engine.synthesize({'source': baseline['source_reference'], 'text': baseline['expected'],
                            'language': baseline['language'], 'reference_text': baseline['result']['reference_text']},
                            folder, progress, prepared)
                        trial['state'] = 'generated'
                    except Exception as exc:
                        trial.update(state='failed', error=f'{type(exc).__name__}: {exc}')
                        save()
                        raise
                    save()
        del engine
        gc.collect()
        for trial in report['trials']:
            result = trial['result']
            folder = root/trial['name']
            trial['audit'] = full_audit(result['reference_preview'], result['audio'], baseline['expected'],
                                       folder, baseline['language'], progress)
            trial['state'] = 'measured'
            save()
            print(json.dumps({'name':trial['name'],'synthesis_seconds':result['synthesis_seconds'],
                              'speaker_cosine':trial['audit']['speaker_verification'].get('cosine_similarity'),
                              'content':trial['audit']['content_comparison']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
