"""Measure original-ONNX CPU decoder threading on previously captured English inputs."""
import argparse
import gc
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', choices=[1, 2, 4], default=[2, 4, 1])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    captured = json.loads(args.capture_report.read_text(encoding='utf-8'))
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    graph = download_models()['conditional_decoder']
    report = {'capture_report': str(args.capture_report), 'status': 'running',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Original model, CPU provider and captured inputs. Internal random noise is not frozen.',
              'trials': []}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(args.output)
    progress = lambda message, **kwargs: print(message, flush=True)
    with keep_awake():
        for threads in args.threads:
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            decoder = ort.InferenceSession(graph, options, providers=['CPUExecutionProvider'])
            trial = {'threads': threads, 'runs': [], 'state': 'running'}
            report['trials'].append(trial)
            save()
            for repeat in range(3):
                waveforms, times = [], []
                for item in captured['captured_passages']:
                    with np.load(item['feeds'], allow_pickle=False) as archive:
                        feeds = {key: archive[key] for key in archive.files}
                    started = active_time()
                    waveform = decoder.run(None, feeds)[0].reshape(-1)
                    times.append(active_time()-started)
                    if not np.isfinite(waveform).all():
                        raise ValueError('The CPU decoder produced nonfinite audio')
                    waveforms.append(waveform)
                audio = save_audio(folder/f'threads-{threads}-repeat-{repeat+1}.wav', waveforms)
                trial['runs'].append({'audio': audio, 'seconds': sum(times), 'warmup': repeat == 0})
                save()
                print('THREADS', threads, 'REPEAT', repeat+1, 'SECONDS', sum(times), flush=True)
            del decoder, feeds, waveform, waveforms
            gc.collect()
            trial['warm_mean_seconds'] = float(np.mean([run['seconds'] for run in trial['runs'][1:]]))
            trial['state'] = 'generated'
            save()
        for trial in report['trials']:
            destination = folder/f"audit-{trial['threads']}"
            destination.mkdir()
            trial['audit'] = full_audit(captured['cpu_result']['reference_preview'], trial['runs'][-1]['audio'],
                                        captured['expected'], destination, 'en', progress)
            trial['state'] = 'measured'
            save()
        report['status'] = 'measured'
        save()


if __name__ == '__main__':
    main()
