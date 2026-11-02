"""Measure decoder randomness with fixed captured tokens and speaker features.

Repeated seeds test session reproducibility; a separate invocation is needed to
establish cross-process reproducibility. No app seed or defaults are changed.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.models import download_models
from studio.runtime import active_time, keep_awake
from studio.recognition import full_audit
from scripts.english_decoder_trial import save_audio
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-report', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[666, 666, 777])
    parser.add_argument('--skip-audit', action='store_true', help='For a separate-process WAV-hash control only.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    captured = json.loads(args.capture_report.read_text(encoding='utf-8'))
    result = captured.get('result', captured.get('cpu_result'))
    if not result or not captured.get('captured_passages'):
        parser.error('Use a completed real decoder-input capture.')
    baseline = json.loads(Path(captured['baseline_report']).read_text(encoding='utf-8'))
    expected = captured.get('expected', baseline['expected'])
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    graph = download_models()['conditional_decoder']
    report = {'status': 'running', 'capture_report': str(args.capture_report), 'threads': 4,
              'reference_sha256': hashlib.sha256(Path(result['reference_preview']).read_bytes()).hexdigest(),
              'scope': 'Same speech tokens/features, original CPU decoder, explicit global ORT seed. No app change or human listening claim.',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'trials': []}
    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    save()
    with keep_awake():
        for index, seed in enumerate(args.seeds):
            trial = {'seed': seed, 'state': 'running'}
            report['trials'].append(trial)
            save()
            ort.set_seed(seed)
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            decoder = ort.InferenceSession(graph, options, providers=['CPUExecutionProvider'])
            waveforms, times, input_hashes = [], [], []
            try:
                for passage in captured['captured_passages']:
                    with np.load(passage['feeds'], allow_pickle=False) as archive:
                        feeds = {key: archive[key] for key in archive.files}
                    before = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in feeds.items()}
                    started = active_time()
                    waveform = decoder.run(None, feeds)[0].reshape(-1)
                    times.append(active_time()-started)
                    if not np.isfinite(waveform).all():
                        raise ValueError('Nonfinite decoder output')
                    if before != {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in feeds.items()}:
                        raise ValueError('Decoder inputs were modified')
                    input_hashes.append(before)
                    waveforms.append(waveform.copy())
                audio = save_audio(folder/f'seed-{seed}-run-{index+1}.wav', waveforms)
                trial.update(state='decoded', audio=audio, decoder_seconds=sum(times), input_hashes=input_hashes,
                             audio_sha256=hashlib.sha256(Path(audio).read_bytes()).hexdigest())
                print('SEED', seed, 'SECONDS', sum(times), 'SHA', trial['audio_sha256'], flush=True)
            except Exception as exc:
                trial.update(state='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                del decoder
                gc.collect()
                save()
        if not args.skip_audit:
            audits = {}
            if captured.get('audit'):
                capture_digest = hashlib.sha256(Path(result['audio']).read_bytes()).hexdigest()
                audits[capture_digest] = ('capture-report', captured['audit'])
            for index, trial in enumerate(report['trials']):
                digest = trial['audio_sha256']
                if digest in audits:
                    trial['audit_reused_from'] = audits[digest][0]
                    trial['audit'] = audits[digest][1]
                else:
                    destination = folder/f'audit-{index+1}'
                    destination.mkdir()
                    trial['audit'] = full_audit(result['reference_preview'], trial['audio'], expected,
                                               destination, 'en', lambda message: print(message, flush=True))
                    audits[digest] = (index+1, trial['audit'])
                trial['state'] = 'measured'
                save()
        report['status'] = 'decoded' if args.skip_audit else 'measured'
        report['repeated_seed_hashes'] = {str(seed): sorted({trial['audio_sha256'] for trial in report['trials']
                                                          if trial['seed'] == seed}) for seed in args.seeds}
        save()


if __name__ == '__main__':
    main()
