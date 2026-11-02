"""Render and audit real English speech with the cached Q8 LM on OpenVINO CPU.

The voice encoder, token embedding and waveform decoder remain on original ORT.
This isolated experiment does not change app defaults or freeze vocoder noise.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.models import download_models, PRECISION
from studio.speech import SpeechEngine, prepare_sample
from studio.runtime import active_time, keep_awake
from studio.recognition import full_audit
from studio.analysis import analyze_files, normalize_text
from scripts.english_decoder_trial import bundled_decoder, CapturingDecoder
import numpy as np


def cpu_language_config(execution_mode='performance'):
    config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32',
              'INFERENCE_NUM_THREADS': 2, 'EXECUTION_MODE_HINT': execution_mode.upper()}
    if execution_mode == 'accuracy':
        config.update(DYNAMIC_QUANTIZATION_GROUP_SIZE=0, KV_CACHE_PRECISION='f32')
    return config


class CpuLanguage:
    def __init__(self, graph, progress, execution_mode='performance'):
        import openvino as ov
        self.core = ov.Core()
        started = active_time()
        progress('Preparing the unchanged Q8 language model on OpenVINO CPU')
        model = self.core.read_model(bundled_decoder(graph))
        self.inputs = []
        for port in model.inputs:
            dtype = np.dtype(port.get_element_type().to_dtype())
            self.inputs.append(SimpleNamespace(name=port.get_any_name(),
                shape=[dim.get_length() if dim.is_static else None for dim in port.get_partial_shape()],
                type='tensor(float16)' if dtype == np.float16 else 'tensor(float)'))
        self.names = ['logits']+[item.name.replace('past_key_values.', 'present.', 1)
                                 for item in self.inputs if 'past_key_values' in item.name]
        self.compiled = self.core.compile_model(model, 'CPU', cpu_language_config(execution_mode))
        self.request = self.compiled.create_infer_request()
        for name in self.names:
            self.compiled.output(name)
        self.compile_seconds = active_time()-started
        self.runtime_version = ov.__version__
        self.calls, self.inference_seconds = 0, 0.0

    def get_inputs(self):
        return self.inputs

    def run(self, outputs, feeds):
        started = active_time()
        self.request.infer(feeds)
        values = [self.request.get_tensor(name).data.copy() for name in self.names]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError('The candidate language model produced nonfinite output')
        self.inference_seconds += active_time()-started
        self.calls += 1
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execution-mode', choices=['performance', 'accuracy'], default='performance')
    parser.add_argument('--decoder-seed', type=int, help='Set and record the global ORT seed before constructing sessions.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    if baseline['language'] != 'en':
        parser.error('Use an English baseline.')
    paths = download_models()
    graph = Path(paths['language_model']).with_name('language_model_quantized.onnx')
    if not graph.is_file() or not Path(str(graph)+'_data').is_file():
        parser.error('The Q8 weights must already be cached.')
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    prepared = prepare_sample(baseline['source_reference'], folder)
    if hashlib.sha256(prepared[1].read_bytes()).digest() != hashlib.sha256(Path(baseline['result']['reference_preview']).read_bytes()).digest():
        raise ValueError('The selected reference changed')
    report = {'status': 'running', 'baseline_report': str(args.baseline_report),
              'scope': 'Experimental Q8 language backend with original encoders/decoder. No perceptual-equivalence claim.',
              'decoder_randomness': 'Internal vocoder noise; requested global seed is recorded. PCM identity requires a direct check.',
              'requested_ort_seed': args.decoder_seed,
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(args.output)
    progress = lambda message, **kwargs: print(message, flush=True)
    save()
    engine = None
    try:
        with keep_awake():
            try:
                if args.decoder_seed is not None:
                    import onnxruntime as ort
                    ort.set_seed(args.decoder_seed)
                engine = SpeechEngine(progress)
                engine.sessions.pop('language_model')
                gc.collect()
                engine.sessions['language_model'] = CpuLanguage(graph, progress, args.execution_mode)
                engine.sessions['conditional_decoder'] = CapturingDecoder(engine.sessions['conditional_decoder'], folder)
                engine.precision = {**PRECISION, 'language_model': 'quantized',
                    'language_backend': 'openvino-cpu', 'float_precision_hint': 'f32', 'language_threads': 2}
                engine.precision['execution_mode'] = args.execution_mode
                report['result'] = engine.synthesize({'source': baseline['source_reference'],
                    'text': baseline['expected'], 'language': 'en'}, folder, progress, prepared=prepared)
                report['language_runtime'] = {
                    'version': engine.sessions['language_model'].runtime_version,
                    'compile_seconds': engine.sessions['language_model'].compile_seconds,
                    'inference_seconds': engine.sessions['language_model'].inference_seconds,
                    'calls': engine.sessions['language_model'].calls}
                report['captured_passages'] = engine.sessions['conditional_decoder'].calls
                report['state_before_audit'] = 'generated'
                report['status'] = 'generated'
                save()
            finally:
                engine = None
                gc.collect()
            result = report['result']
            report['audit'] = full_audit(result['reference_preview'], result['audio'], baseline['expected'], folder, 'en', progress)
            if normalize_text(baseline.get('reference_text', '')) == normalize_text(baseline['expected']):
                report['same_words_acoustics'], _, _, _ = analyze_files(
                    result['reference_preview'], result['audio'], folder/'same-words', same_words=True)
            report['status'] = 'measured'
            save()
            print(json.dumps({'seconds': result['synthesis_seconds'],
                'speaker': report['audit']['speaker_verification'].get('cosine_similarity'),
                'content': report['audit']['content_comparison']}, ensure_ascii=False), flush=True)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
