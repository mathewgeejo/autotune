"""Optional full-precision Intel flow acceleration with a CPU fallback."""
import gc
import importlib
import os

import numpy as np

from studio.audio import AudioError
from studio.paths import MODELS
from studio.runtime import active_time


class WarmShapeFlow:
    """Warm a new GPU shape without advancing or modifying the solver input.

    Dynamic GPU kernels can change after their first evaluation. Three repeated
    evaluations reproduced the established warm reading in two fresh processes.
    Ordinary CPU inference is left alone, including a fallback during warm-up.
    """
    def __init__(self, flow, evaluations=3):
        if evaluations < 1:
            raise ValueError('At least one evaluation is required')
        self.flow, self.evaluations, self.shapes = flow, evaluations, set()
        self.warmup_seconds = 0.0
        self.extra_evaluations = 0

    @property
    def backend(self):
        return getattr(self.flow, 'backend', 'onnxruntime-cpu')

    def run(self, output_names, feeds):
        shape = tuple(feeds['x'].shape)
        if shape not in self.shapes and self.backend.startswith('openvino'):
            started = active_time()
            self.shapes.add(shape)
            for _ in range(self.evaluations-1):
                value = self.flow.run(output_names, feeds)
                if not self.backend.startswith('openvino'):
                    self.warmup_seconds += active_time()-started
                    return value
                self.extra_evaluations += 1
            self.warmup_seconds += active_time()-started
        return self.flow.run(output_names, feeds)


class IntelFlowSession:
    backend = 'openvino-intel-gpu-fp32'

    def __init__(self, path, progress, cpu_factory, openvino):
        self.path, self.progress, self.cpu_factory = path, progress, cpu_factory
        self.cpu = None
        self.core = openvino.Core()
        if 'GPU' not in self.core.available_devices:
            raise RuntimeError('No compatible Intel GPU is available')
        cache = MODELS/'intel-gpu-cache'
        cache.mkdir(parents=True, exist_ok=True)
        progress('Preparing full-precision Intel GPU acceleration')
        started = active_time()
        self.compiled = self.core.compile_model(str(path), 'GPU', {
            'PERFORMANCE_HINT':'LATENCY', 'INFERENCE_PRECISION_HINT':'f32', 'CACHE_DIR':str(cache)})
        self.compile_seconds = active_time()-started
        self.request = self.compiled.create_infer_request()

    def run(self, output_names, feeds):
        if self.cpu is not None:
            return self.cpu.run(output_names, feeds)
        try:
            # InferRequest copies inputs by default. Copy the output too: its
            # storage is reused by the next Euler step.
            self.request.infer(feeds)
            value = self.request.get_output_tensor(0).data.copy()
            if value.shape != feeds['x'].shape or not np.isfinite(value).all():
                raise RuntimeError('Invalid Intel GPU flow output')
            return [value]
        except RuntimeError:
            self.progress('Intel GPU inference stopped; continuing the same refinement step on CPU')
            self.request = self.compiled = self.core = None
            gc.collect()
            self.cpu = self.cpu_factory(self.path)
            self.backend = 'onnxruntime-cpu-after-intel-fallback'
            return self.cpu.run(output_names, feeds)


def flow_session(path, progress, cpu_factory):
    mode = os.environ.get('VOICE_STUDIO_ACCELERATOR', 'auto').strip().lower()
    if mode not in ('auto', 'cpu', 'intel'):
        raise AudioError('Choose auto, cpu, or intel for VOICE_STUDIO_ACCELERATOR.')
    if mode == 'cpu':
        return cpu_factory(path)
    try:
        openvino = importlib.import_module('openvino')
        return IntelFlowSession(path, progress, cpu_factory, openvino)
    except (ImportError, RuntimeError, OSError) as exc:
        if mode == 'intel':
            progress(f'Intel acceleration is unavailable ({type(exc).__name__}); using the CPU')
        return cpu_factory(path)
