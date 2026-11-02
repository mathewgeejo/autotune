"""Probe the unchanged cached Q8 English language model on OpenVINO CPU.

Uses identical prefill/incremental feeds on both runtimes. This stage-only probe
produces no speech and does not change the production backend.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, REFERENCES
configure()
from studio.models import download_models
from studio.speech import prepare_sample, START_TOKEN
from studio.runtime import active_time, keep_awake
from scripts.english_decoder_trial import bundled_decoder
from scripts.english_kv_padding import pad_incremental_cache, compact_present
from scripts.english_openvino_trial import cpu_language_config
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=[1, 3], default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-capacity', type=int, choices=[0, 512, 1024], default=0)
    parser.add_argument('--execution-mode', choices=['performance', 'accuracy'], default='performance')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    prepared = prepare_sample(baseline['source_reference'], folder)
    if hashlib.sha256(prepared[1].read_bytes()).digest() != hashlib.sha256(Path(baseline['result']['reference_preview']).read_bytes()).digest():
        raise ValueError('The selected conditioning reference changed')
    with np.load(REFERENCES/prepared[2]/'conditioning.npz', allow_pickle=False) as cached:
        cond = cached['cond'].copy()
    paths = download_models()
    graph = Path(paths['language_model']).with_name('language_model_quantized.onnx')
    if not graph.is_file() or not Path(str(graph)+'_data').is_file():
        parser.error('The Q8 model is not cached; this probe does not download it.')
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    report = {'status': 'running', 'scope': 'Identical-input Q8 language-model probe; no speech output or app backend change.',
              'requested_backend': 'openvino-cpu-fp32', 'threads': 2,
              'cache_capacity': args.cache_capacity, 'execution_mode': args.execution_mode,
              'requested_properties': cpu_language_config(args.execution_mode), 'steps': []}
    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    try:
        with keep_awake():
            embedding = ort.InferenceSession(paths['embed_tokens'], options, providers=['CPUExecutionProvider'])
            tokenizer = Tokenizer.from_file(paths['tokenizer'])
            ids = np.array([tokenizer.encode(baseline['expected']).ids], dtype=np.int64)
            embeds = np.concatenate((cond, embedding.run(None, {'input_ids': ids})[0]), axis=1)
            source = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
            inputs = source.get_inputs()
            names = [output.name for output in source.get_outputs()]
            batch, length, _ = embeds.shape
            past = {}
            for item in inputs:
                if 'past_key_values' in item.name:
                    dtype = np.float16 if item.type == 'tensor(float16)' else np.float32
                    past[item.name] = np.zeros((batch, item.shape[1], 0, item.shape[3]), dtype=dtype)
            feeds = {'inputs_embeds': embeds, 'attention_mask': np.ones((batch, length), dtype=np.int64),
                     'position_ids': np.arange(length, dtype=np.int64).reshape(1, -1), **past}
            import openvino as ov
            report['runtime_version'] = ov.__version__
            bundled = bundled_decoder(graph)
            started = active_time()
            core = ov.Core()
            compiled = core.compile_model(bundled, 'CPU', cpu_language_config(args.execution_mode))
            report['compile_seconds'] = active_time()-started
            request = compiled.create_infer_request()
            unpadded_request = compiled.create_infer_request() if args.cache_capacity else None
            tokens = [START_TOKEN]
            for step in range(args.steps):
                started = active_time()
                expected = source.run(None, feeds)
                cpu_seconds = active_time()-started
                padded_feeds, restore = pad_incremental_cache(feeds, args.cache_capacity)
                unpadded_values = None
                if restore is not None:
                    unpadded_request.infer(feeds)
                    unpadded_values = [unpadded_request.get_tensor(name).data.copy() for name in names]
                runs = []
                for repeat in range(2):
                    started = active_time()
                    request.infer(padded_feeds)
                    raw_values = [request.get_tensor(name).data.copy() for name in names]
                    if restore is not None and repeat == 0:
                        report['padding_shape_diagnostic'] = {'history': restore[0], 'capacity': restore[1],
                            'outputs': {name: list(value.shape) for name, value in zip(names, raw_values)}}
                        save()
                    actual = compact_present(raw_values, restore)
                    runs.append(active_time()-started)
                differences = []
                for name, before, after in zip(names, expected, actual):
                    if before.shape != after.shape or not np.isfinite(after).all():
                        raise ValueError(f'Invalid candidate output: {name}')
                    differences.append({'name': name, 'max_absolute_error': float(np.max(np.abs(before-after))),
                        'relative_norm_error': float(np.linalg.norm(before-after)/max(np.linalg.norm(before), 1e-9))})
                def next_token(value):
                    scores = value[:, -1, :].copy()
                    repeated = np.unique(tokens)
                    penalized = scores[:, repeated]
                    scores[:, repeated] = np.where(penalized < 0, penalized*1.2, penalized/1.2)
                    return int(np.argmax(scores, axis=-1)[0])
                token, candidate_token = next_token(expected[0]), next_token(actual[0])
                report['steps'].append({'step': step, 'onnxruntime_seconds': cpu_seconds,
                    'openvino_seconds': runs, 'onnxruntime_token': token, 'openvino_token': candidate_token,
                    'same_token': token == candidate_token, 'outputs': differences})
                if unpadded_values is not None:
                    report['steps'][-1]['padding_comparison'] = {
                        'same_token_as_unpadded': next_token(unpadded_values[0]) == candidate_token,
                        'logits_max_absolute_error': float(np.max(np.abs(unpadded_values[0]-actual[0]))),
                        'logits_close': bool(np.allclose(unpadded_values[0], actual[0], rtol=1e-5, atol=1e-5)),
                        'all_output_shapes_match': all(a.shape == b.shape for a, b in zip(unpadded_values, actual))}
                print('STEP', step, 'ORT', cpu_seconds, 'OV', runs, 'SAME TOKEN', token == candidate_token, flush=True)
                save()
                tokens.append(token)
                embeds = embedding.run(None, {'input_ids': np.array([[token]], dtype=np.int64)})[0]
                feeds = {'inputs_embeds': embeds,
                         'attention_mask': np.concatenate((feeds['attention_mask'], np.ones((batch, 1), dtype=np.int64)), axis=1),
                         'position_ids': feeds['position_ids'][:, -1:]+1,
                         **dict(zip(past, expected[1:]))}
            report['status'] = 'measured'
            report['all_probe_tokens_match'] = all(step['same_token'] for step in report['steps'])
            del request, compiled, core, source, embedding
            gc.collect()
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
