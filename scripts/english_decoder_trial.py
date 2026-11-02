"""Compare the English CPU and Intel waveform decoders using identical captured feeds.

This experiment does not change production backend selection. Repeated CPU runs
calibrate any stochastic variation inside the exported decoder before GPU errors
are interpreted. Other synthesis models are released before GPU compilation.
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
from studio.paths import configure, MODELS
configure()
from studio.models import download_models
from studio.recognition import full_audit
from studio.runtime import active_time, keep_awake
from studio.speech import SpeechEngine, prepare_sample, RATE
import numpy as np
import soundfile as sf


def waveform_comparison(candidate, reference):
    if candidate.shape != reference.shape:
        raise ValueError(f'Decoder waveform lengths differ: {candidate.shape} / {reference.shape}')
    if not np.isfinite(candidate).all():
        raise ValueError('Decoder returned a nonfinite waveform')
    return {'relative_rms_error': float(np.linalg.norm(candidate-reference)/max(np.linalg.norm(reference), 1e-9)),
            'correlation': float(np.corrcoef(candidate, reference)[0, 1])}


def save_audio(path, waveforms):
    # Match production's per-passage peak limiting before concatenation.
    audio = np.concatenate([waveform*min(1.0, 0.98/max(float(np.max(np.abs(waveform))), 0.001))
                            for waveform in waveforms])
    sf.write(path, audio, RATE, subtype='PCM_16')
    return str(path)


class CapturingDecoder:
    def __init__(self, session, folder):
        self.session, self.folder, self.calls = session, folder, []

    def run(self, outputs, feeds):
        name = f'passage-{len(self.calls)+1}'
        copied = {key: value.copy() for key, value in feeds.items()}
        path = self.folder/(name+'-feeds.npz')
        np.savez(path, **copied)
        started = active_time()
        result = self.session.run(outputs, feeds)
        seconds = active_time()-started
        waveform = result[0].reshape(-1).copy()
        np.save(self.folder/(name+'-cpu.npy'), waveform)
        self.calls.append({'name': name, 'feeds': str(path), 'seconds': seconds,
                           'input_shapes': {key: list(value.shape) for key, value in copied.items()}})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--text-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path to preserve prior measurements.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    if baseline['language'] != 'en':
        parser.error('Choose an English baseline.')
    text = args.text_file.read_text(encoding='utf-8') if args.text_file else baseline['expected']
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    reference_folder = folder/'reference'
    reference_folder.mkdir()
    prepared = prepare_sample(baseline['source_reference'], reference_folder)
    assert hashlib.sha256(prepared[1].read_bytes()).digest() == hashlib.sha256(Path(baseline['result']['reference_preview']).read_bytes()).digest(), 'Selected reference differs from baseline'
    report = {'status': 'running', 'baseline_report': str(args.baseline_report), 'expected': text,
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Identical decoder weights and captured inputs; English production backend remains unchanged.',
              'cpu_repeats': [], 'intel_repeats': []}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(args.output)
    progress = lambda message, **kwargs: print(message, flush=True)
    save()
    with keep_awake():
        try:
            engine = SpeechEngine(progress)
            capture = CapturingDecoder(engine.sessions['conditional_decoder'], folder)
            engine.sessions['conditional_decoder'] = capture
            cpu_folder = folder/'cpu'
            cpu_folder.mkdir()
            result = engine.synthesize({'source': baseline['source_reference'], 'text': text, 'language': 'en'},
                                       cpu_folder, progress, prepared)
            report.update(cpu_result=result, captured_passages=capture.calls)
            save()
            # Retain only this decoder for repeated-CPU controls.
            engine.sessions.clear()
            del engine
            gc.collect()
            for repeat in range(args.repeats):
                waveforms, metrics = [], []
                for item in capture.calls:
                    with np.load(item['feeds'], allow_pickle=False) as archive:
                        feeds = {key: archive[key] for key in archive.files}
                    reference = np.load(folder/(item['name']+'-cpu.npy'), allow_pickle=False)
                    started = active_time()
                    waveform = capture.session.run(None, feeds)[0].reshape(-1).copy()
                    seconds = active_time()-started
                    metrics.append({'seconds': seconds, **waveform_comparison(waveform, reference)})
                    waveforms.append(waveform)
                audio = save_audio(folder/f'cpu-repeat-{repeat+1}.wav', waveforms)
                report['cpu_repeats'].append({'audio': audio, 'passages': metrics})
                save()
                print('CPU repeat', repeat+1, metrics, flush=True)
            calls = capture.calls
            del capture, feeds, reference, waveform, waveforms
            gc.collect()
            import openvino as ov
            core = ov.Core()
            if 'GPU' not in core.available_devices:
                raise RuntimeError('No Intel GPU is available')
            graph = download_models(progress)['conditional_decoder']
            cache = MODELS/'intel-english-decoder-cache'
            cache.mkdir(exist_ok=True)
            progress('Compiling the unchanged English waveform decoder for Intel GPU')
            started = active_time()
            compiled = core.compile_model(graph, 'GPU', {'PERFORMANCE_HINT': 'LATENCY',
                'INFERENCE_PRECISION_HINT': 'f32', 'CACHE_DIR': str(cache)})
            report.update(compile_seconds=active_time()-started, openvino_version=ov.get_version())
            request = compiled.create_infer_request()
            save()
            for repeat in range(args.repeats):
                waveforms, metrics = [], []
                for item in calls:
                    with np.load(item['feeds'], allow_pickle=False) as archive:
                        feeds = {key: archive[key] for key in archive.files}
                    reference = np.load(folder/(item['name']+'-cpu.npy'), allow_pickle=False)
                    started = active_time()
                    request.infer(feeds)
                    waveform = request.get_output_tensor(0).data.reshape(-1).copy()
                    seconds = active_time()-started
                    metrics.append({'seconds': seconds, **waveform_comparison(waveform, reference)})
                    waveforms.append(waveform)
                audio = save_audio(folder/f'intel-repeat-{repeat+1}.wav', waveforms)
                report['intel_repeats'].append({'audio': audio, 'passages': metrics})
                save()
                print('Intel repeat', repeat+1, metrics, flush=True)
            del request, compiled, core, feeds, waveform, reference, waveforms
            gc.collect()
            report['status'] = 'generated'
            save()
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save()
            raise
        for name, audio in [('cpu', result['audio']), ('intel', report['intel_repeats'][-1]['audio'])]:
            destination = folder/(name+'-audit')
            destination.mkdir()
            report[name+'_audit'] = full_audit(result['reference_preview'], audio, text, destination, 'en', progress)
            save()
        report['status'] = 'measured'
        save()
        print('Measured report:', args.output, flush=True)


if __name__ == '__main__':
    main()
