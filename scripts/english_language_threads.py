"""Time English autoregression while requiring exact equality of decoder inputs.

The decoder is a timing stub in this stage-only experiment. No new speech audio
is claimed. Every generated speech token, speaker embedding, and speaker feature
must match a previously measured real synthesis before a setting is considered.
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
from studio.runtime import keep_awake
from studio.speech import SpeechEngine, prepare_sample
import numpy as np
import onnxruntime as ort


class DecoderInputCheck:
    def __init__(self, expected):
        self.expected = expected
        self.last = None

    def run(self, outputs, feeds):
        self.last = {'all_inputs_identical': all(np.array_equal(value, self.expected[key])
                                                for key, value in feeds.items()),
                     'speech_tokens_sha256': hashlib.sha256(feeds['speech_tokens'].tobytes()).hexdigest(),
                     'speech_token_count_including_prompt': feeds['speech_tokens'].shape[1]}
        return [np.full(24000, 0.1, dtype=np.float32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', choices=[1, 2, 4], default=[2, 4, 1])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path to retain prior results.')
    captured = json.loads(args.capture_report.read_text(encoding='utf-8'))
    if len(captured['captured_passages']) != 1:
        parser.error('Choose a single-passage capture.')
    baseline = json.loads(Path(captured['baseline_report']).read_text(encoding='utf-8'))
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    prepared = prepare_sample(baseline['source_reference'], folder)
    if hashlib.sha256(prepared[1].read_bytes()).digest() != hashlib.sha256(Path(captured['cpu_result']['reference_preview']).read_bytes()).digest():
        raise ValueError('The selected voice reference changed')
    with np.load(captured['captured_passages'][0]['feeds'], allow_pickle=False) as archive:
        expected = {key: archive[key] for key in archive.files}
    report = {'capture_report': str(args.capture_report), 'status': 'running', 'trials': [],
              'scope': 'Autoregression-only timing; stub decoder creates no experimental speech output. Exact real-decoder input equality is required.',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output)
    progress = lambda message, **kwargs: print(message, flush=True)
    with keep_awake():
        engine = SpeechEngine(progress)
        voice, _ = engine.conditioning(prepared[0], prepared[2], progress)
        del engine.sessions['conditional_decoder'], engine.sessions['speech_encoder'], engine.sessions['language_model']
        gc.collect()
        check = DecoderInputCheck(expected)
        engine.sessions['conditional_decoder'] = check
        graph = download_models(progress)['language_model']
        for threads in args.threads:
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            engine.sessions['language_model'] = ort.InferenceSession(graph, options, providers=['CPUExecutionProvider'])
            trial = {'threads': threads, 'runs': []}
            report['trials'].append(trial)
            save()
            for repeat in range(3):
                engine.generate_chunk(captured['expected'], voice, progress)
                measured = {'seconds': engine.last_chunk_metrics['language_model_seconds'],
                            'warmup': repeat == 0, **check.last}
                trial['runs'].append(measured)
                save()
                print('LANGUAGE THREADS', threads, measured, flush=True)
            trial['warm_mean_seconds'] = float(np.mean([item['seconds'] for item in trial['runs'][1:]]))
            del engine.sessions['language_model']
            gc.collect()
            save()
        report['status'] = 'measured'
        save()


if __name__ == '__main__':
    main()
