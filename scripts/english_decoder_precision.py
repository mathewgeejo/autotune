"""Compare Q8/FP32 decoders with identical captured inputs and internal noise.

Uses existing pinned weights only. Derived graphs hard-link external weights;
the Hugging Face cache and application defaults are never modified.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
sys.path.append(str(ROOT/'.tools/model-tools'))
import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

from scripts.english_decoder_trial import save_audio, waveform_comparison
from studio.models import download_models, REVISION
from studio.recognition import full_audit
from studio.runtime import active_time, keep_awake


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def random_nodes(model):
    nodes = [node for node in model.graph.node if node.op_type.startswith('Random')]
    if len(nodes) != 4 or any(node.op_type not in ('RandomNormalLike', 'RandomUniformLike') for node in nodes):
        raise ValueError('The pinned decoder random operations changed.')
    return nodes


def derive(graph, destination, capture=False):
    model = onnx.load(str(graph), load_external_data=False)
    nodes = random_nodes(model)
    names = [name for node in nodes for name in node.output]
    info = {value.name: value for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
    for name in names:
        if name not in info or info[name].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError('Random tensor type/shape metadata is missing.')
        value = helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        value.CopyFrom(info[name])
        (model.graph.output if capture else model.graph.input).append(value)
    if not capture:
        kept = [node for node in model.graph.node if node.op_type not in ('RandomNormalLike', 'RandomUniformLike')]
        del model.graph.node[:]
        model.graph.node.extend(kept)
    locations = {item.value for tensor in model.graph.initializer for item in tensor.external_data
                 if item.key == 'location'}
    for location in locations:
        if Path(location).name != location:
            raise ValueError('External weights must use a local filename.')
        source = (graph.parent/location).resolve(strict=True)
        target = destination.parent/location
        if not target.exists():
            os.link(source, target)
        if not os.path.samefile(source, target):
            raise ValueError('Derived graph weights must be hard links to the pinned source.')
    onnx.save_model(model, str(destination))
    return names


def session(graph):
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-report', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[666])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    captured = json.loads(args.capture_report.read_text(encoding='utf-8'))
    result = captured['cpu_result']
    if result['model_revision'] != REVISION or not captured['captured_passages']:
        parser.error('Use a completed capture from the pinned English model.')
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    graphs = folder/'graphs'
    graphs.mkdir()
    q8 = Path(download_models()['conditional_decoder'])
    fp32 = q8.with_name('conditional_decoder.onnx')
    if not fp32.is_file() or not Path(str(fp32)+'_data').is_file():
        parser.error('The FP32 decoder must already be cached; this script does not download it.')
    capture_graph, q8_graph, fp32_graph = [graphs/name for name in ('capture.onnx', 'q8-replay.onnx', 'fp32-replay.onnx')]
    names = derive(q8, capture_graph, capture=True)
    if derive(q8, q8_graph) != names or derive(fp32, fp32_graph) != names:
        raise ValueError('Decoder random tensors do not correspond.')
    report = {'status': 'running', 'capture_report': str(args.capture_report),
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Only waveform-decoder precision changes. Tokens, speaker features and four internal noise tensors match. No production change or perceptual identity claim.',
              'reference_sha256': hashlib.sha256(Path(result['reference_preview']).read_bytes()).hexdigest(),
              'random_outputs': names, 'trials': []}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(args.output)
    def progress(message):
        print(message, flush=True)
    save()
    with keep_awake():
        try:
            for seed in args.seeds:
                trial = {'seed': seed, 'state': 'capturing', 'passages': [], 'variants': {}}
                report['trials'].append(trial)
                destination = folder/f'seed-{seed}'
                destination.mkdir()
                save()
                decoder = None
                ort.set_seed(seed)
                waveforms, times = [], []
                try:
                    progress(f'Seed {seed}: capturing Q8 waveform and internal noise')
                    decoder = session(capture_graph)
                    for index, passage in enumerate(captured['captured_passages'], 1):
                        with np.load(passage['feeds'], allow_pickle=False) as archive:
                            feeds = {key: archive[key] for key in archive.files}
                        started = active_time()
                        values = decoder.run(None, feeds)
                        times.append(active_time()-started)
                        waveform = values[0].reshape(-1).copy()
                        noises = dict(zip(names, values[-len(names):]))
                        if not all(np.isfinite(value).all() for value in [waveform, *noises.values()]):
                            raise ValueError('Nonfinite waveform/noise')
                        noise_path = destination/f'passage-{index}-noise.npz'
                        np.savez(noise_path, **noises)
                        np.save(destination/f'passage-{index}-q8.npy', waveform)
                        waveforms.append(waveform)
                        trial['passages'].append({'feeds': passage['feeds'], 'noise': str(noise_path),
                            'input_hashes': {key: digest(value) for key, value in feeds.items()},
                            'noise_hashes': {key: digest(value) for key, value in noises.items()}})
                        save()
                    trial['variants']['q8'] = {'audio': save_audio(destination/'q8.wav', waveforms),
                                                'decoder_seconds': sum(times)}
                finally:
                    decoder = None
                    gc.collect()
                for name, graph in [('q8-replay', q8_graph), ('fp32', fp32_graph)]:
                    progress(f'Seed {seed}: replaying {name} with the same internal noise')
                    trial['state'] = name
                    save()
                    waveforms, metrics = [], []
                    try:
                        decoder = session(graph)
                        for index, passage in enumerate(trial['passages'], 1):
                            with np.load(passage['feeds'], allow_pickle=False) as archive:
                                feeds = {key: archive[key] for key in archive.files}
                            with np.load(passage['noise'], allow_pickle=False) as archive:
                                noises = {key: archive[key] for key in archive.files}
                            if passage['input_hashes'] != {key: digest(value) for key, value in feeds.items()}:
                                raise ValueError('Captured voice/token inputs changed.')
                            if passage['noise_hashes'] != {key: digest(value) for key, value in noises.items()}:
                                raise ValueError('Captured random noise changed.')
                            started = active_time()
                            waveform = decoder.run(None, {**feeds, **noises})[0].reshape(-1).copy()
                            seconds = active_time()-started
                            reference = np.load(destination/f'passage-{index}-q8.npy', allow_pickle=False)
                            comparison = waveform_comparison(waveform, reference)
                            if name == 'q8-replay' and comparison['relative_rms_error'] > 1e-5:
                                raise ValueError(f'The Q8 noise-replay control failed: {comparison}')
                            metrics.append({'seconds': seconds, **comparison})
                            waveforms.append(waveform)
                        trial['variants'][name] = {'audio': save_audio(destination/f'{name}.wav', waveforms),
                            'decoder_seconds': sum(item['seconds'] for item in metrics), 'passages': metrics}
                    finally:
                        decoder = None
                        gc.collect()
                        save()
                trial['state'] = 'decoded'
                save()
                for name in ('q8', 'fp32'):
                    progress(f'Seed {seed}: independently auditing {name}')
                    audit_folder = destination/f'audit-{name}'
                    audit_folder.mkdir()
                    variant = trial['variants'][name]
                    variant['audit'] = full_audit(result['reference_preview'], variant['audio'], captured['expected'],
                                                   audit_folder, 'en', progress)
                    save()
                trial['state'] = 'measured'
                save()
            report['status'] = 'measured'
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save()
    print('Matched-noise decoder precision comparison completed.', flush=True)


if __name__ == '__main__':
    main()
