"""Compare Q8/FP32 reference encoders with unchanged English text/decoder models."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, REFERENCES
configure()
from studio.models import PRECISION, REVISION
from studio.recognition import full_audit
from studio.runtime import active_time, keep_awake
from studio.speech import SpeechEngine, prepare_sample
from scripts.english_decoder_trial import CapturingDecoder
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--download-report', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[666])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    downloaded = json.loads(args.download_report.read_text(encoding='utf-8'))
    if downloaded['status'] != 'verified' or downloaded['revision'] != REVISION:
        parser.error('The pinned FP32 encoder must have a verified download report.')
    graph = Path(next(item['path'] for item in downloaded['files'] if item['name'].endswith('.onnx')))
    for item in downloaded['files']:
        path = Path(item['path'])
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                digest.update(block)
        if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError('The verified FP32 encoder files changed.')
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    reference_folder = folder/'reference'
    reference_folder.mkdir()
    prepared = prepare_sample(baseline['source_reference'], reference_folder)
    reference_hash = hashlib.sha256(prepared[1].read_bytes()).hexdigest()
    if reference_hash != hashlib.sha256(Path(baseline['result']['reference_preview']).read_bytes()).hexdigest():
        raise ValueError('The selected natural reference changed.')
    cache = REFERENCES/prepared[2]/'conditioning.npz'
    with np.load(cache, allow_pickle=False) as archive:
        q8 = [archive[key] for key in ('cond', 'prompt', 'embedding', 'features')]
    report = {'status': 'running', 'reference_sha256': reference_hash,
              'baseline_report': str(args.baseline_report), 'download_report': str(args.download_report),
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Only the conditioning encoder precision changes. Identical selected recording/text, Q4 text model and Q8 waveform decoder. Paired global seeds do not guarantee matching noise if generated token lengths change.',
              'trials': []}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(args.output)
    def progress(message, **kwargs):
        print(message, flush=True)
    save()
    encoder = None
    with keep_awake():
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            encoder = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
            started = active_time()
            fp32 = encoder.run(None, {'audio_values': prepared[0]})
            report['encoder_seconds'] = active_time()-started
            if len(fp32) != 4 or not all(value.size and np.isfinite(value).all() for value in fp32):
                raise ValueError('FP32 conditioning is incomplete or nonfinite.')
            np.savez(folder/'fp32-conditioning.npz', **dict(zip(('cond', 'prompt', 'embedding', 'features'), fp32)))
            report['conditioning_comparison'] = []
            for name, original, candidate in zip(('cond', 'prompt', 'embedding', 'features'), q8, fp32):
                entry = {'name': name, 'q8_shape': list(original.shape), 'fp32_shape': list(candidate.shape),
                         'identical': np.array_equal(original, candidate)}
                if original.shape == candidate.shape and np.issubdtype(original.dtype, np.floating):
                    entry['relative_rms_error'] = float(np.linalg.norm(original-candidate)/max(np.linalg.norm(original), 1e-9))
                    entry['cosine'] = float(np.dot(original.ravel(), candidate.ravel()) /
                        max(float(np.linalg.norm(original)*np.linalg.norm(candidate)), 1e-9))
                report['conditioning_comparison'].append(entry)
            save()
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save()
            raise
        finally:
            encoder = None
            gc.collect()
        try:
            for seed in args.seeds:
                for name, voice in [('q8', q8), ('fp32', fp32)]:
                    destination = folder/f'{name}-seed-{seed}'
                    destination.mkdir()
                    trial = {'encoder_precision': name, 'seed': seed, 'state': 'rendering'}
                    report['trials'].append(trial)
                    save()
                    engine, capture = None, None
                    try:
                        ort.set_seed(seed)
                        engine = SpeechEngine(progress)
                        engine.conditioning = lambda audio, digest, progress: (voice, False)
                        engine.precision = {**PRECISION, 'speech_encoder': name}
                        capture = CapturingDecoder(engine.sessions['conditional_decoder'], destination)
                        engine.sessions['conditional_decoder'] = capture
                        trial['result'] = engine.synthesize({'source': baseline['source_reference'],
                            'text': baseline['expected'], 'language': 'en',
                            'reference_text': baseline['result'].get('reference_text', '')},
                            destination, progress, prepared=prepared)
                        trial['captured_passages'] = capture.calls
                        trial['state'] = 'generated'
                    finally:
                        capture = None
                        engine = None
                        gc.collect()
                        save()
            for trial in report['trials']:
                destination = Path(trial['result']['audio']).parent/'audit'
                destination.mkdir()
                trial['audit'] = full_audit(prepared[1], trial['result']['audio'], baseline['expected'],
                                             destination, 'en', progress)
                trial['state'] = 'measured'
                save()
            report['status'] = 'measured'
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save()
    print('Reference encoder precision comparison completed.', flush=True)


if __name__ == '__main__':
    main()
