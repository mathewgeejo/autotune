"""Experimental full-precision Intel GPU trial; production remains unchanged."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.tools/openvino-runtime'))
from studio.paths import configure, MODELS
configure()
from studio.indic import IndicEngine
from studio.models import download_indic_models
from studio.recognition import full_audit
from studio.runtime import active_time, keep_awake
from studio.speech import prepare_sample
import numpy as np
import soundfile as sf


class IntelFlow:
    def __init__(self, path, cache):
        import openvino as ov
        self.core = ov.Core()
        if 'GPU' not in self.core.available_devices:
            raise RuntimeError('An Intel GPU is not available to OpenVINO')
        cache.mkdir(parents=True, exist_ok=True)
        started = active_time()
        self.model = self.core.compile_model(str(path), 'GPU', {
            'PERFORMANCE_HINT':'LATENCY', 'INFERENCE_PRECISION_HINT':'f32', 'CACHE_DIR':str(cache)})
        self.compile_seconds = active_time()-started
        self.version = ov.get_version()
        self.request = self.model.create_infer_request()

    def run(self, output_names, feeds):
        self.request.infer(feeds)
        return [self.request.get_output_tensor(0).data.copy()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=True)
    reference = folder/'reference'
    reference.mkdir(exist_ok=True)
    prepared = prepare_sample(baseline['source_reference'], reference)
    assert hashlib.sha256(prepared[1].read_bytes()).digest() == hashlib.sha256(Path(baseline['result']['reference_preview']).read_bytes()).digest()
    progress = lambda message, **kwargs: print(message, flush=True)
    report = {'baseline_report':str(args.baseline_report),'backend':'experimental OpenVINO Intel GPU FP32',
              'scope':'Unchanged model, reference, seed, guidance and step count. No production backend change.', 'trials':[]}
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with keep_awake():
        engine = IndicEngine(progress, 'Detailed')
        engine.flow_decoder = None
        gc.collect()
        root = download_indic_models('Detailed')
        print('Compiling Intel GPU FP32 graph', flush=True)
        flow = IntelFlow(root/'fm_decoder.onnx', MODELS/'intel-gpu-cache')
        engine.flow_decoder = flow
        engine.precision = {**engine.precision, 'flow_backend':'OpenVINO Intel GPU FP32', 'openvino_version':flow.version}
        report.update(compile_seconds=flow.compile_seconds, openvino_version=flow.version)
        save()
        cpu_audio, _ = sf.read(baseline['result']['audio'], dtype='float32')
        for index in range(args.repeats):
            destination = folder/f'run-{index+1}'
            destination.mkdir(exist_ok=True)
            if (destination/'speech.wav').exists():
                raise RuntimeError('Use a fresh output name to retain the prior trial')
            result = engine.synthesize({'source':baseline['source_reference'], 'text':baseline['expected'],
                'language':baseline['language'], 'reference_text':baseline['result']['reference_text']},
                destination, progress, prepared)
            candidate, _ = sf.read(result['audio'], dtype='float32')
            if len(candidate) != len(cpu_audio):
                raise RuntimeError('GPU and CPU output lengths differ')
            error = float(np.linalg.norm(candidate-cpu_audio)/max(np.linalg.norm(cpu_audio),1e-9))
            correlation = float(np.corrcoef(candidate,cpu_audio)[0,1])
            report['trials'].append({'result':result,'cpu_waveform_relative_rms_error':error,
                                    'cpu_waveform_correlation':correlation})
            save()
            print(json.dumps({'run':index+1,'seconds':result['synthesis_seconds'],'relative_rms_error':error,
                              'correlation':correlation}),flush=True)
        del engine, flow
        gc.collect()
        report['audit'] = full_audit(prepared[1], result['audio'], baseline['expected'], folder,
                                    baseline['language'], progress)
        save()
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
