from types import SimpleNamespace

import numpy as np
import pytest

from studio import accelerators


def test_cpu_mode_does_not_import_or_compile_optional_runtime(monkeypatch):
    monkeypatch.setenv('VOICE_STUDIO_ACCELERATOR', 'cpu')
    monkeypatch.setattr(accelerators.importlib, 'import_module', lambda name: (_ for _ in ()).throw(AssertionError('No optional import expected')))
    expected = object()
    assert accelerators.flow_session('model.onnx', lambda message: None, lambda path: expected) is expected


def test_unavailable_gpu_keeps_cpu_generation_available(tmp_path, monkeypatch):
    monkeypatch.setenv('VOICE_STUDIO_ACCELERATOR', 'auto')
    monkeypatch.setattr(accelerators, 'MODELS', tmp_path)
    module = SimpleNamespace(Core=lambda: SimpleNamespace(available_devices=['CPU']))
    monkeypatch.setattr(accelerators.importlib, 'import_module', lambda name: module)
    expected = object()
    assert accelerators.flow_session('model.onnx', lambda message: None, lambda path: expected) is expected


@pytest.mark.parametrize('problem', ['exception', 'nonfinite'])
def test_gpu_output_is_owned_and_failed_step_retries_on_cpu(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(accelerators, 'MODELS', tmp_path)
    outputs = np.array([1.0], dtype=np.float32)
    class Request:
        calls = 0
        def infer(self, feeds):
            self.calls += 1
            if self.calls == 2:
                outputs[:] = np.nan
                if problem == 'exception':
                    raise RuntimeError('Simulated GPU failure')
        def get_output_tensor(self, index):
            return SimpleNamespace(data=outputs)
    request = Request()
    compiled = SimpleNamespace(create_infer_request=lambda: request)
    core = SimpleNamespace(available_devices=['GPU'], compile_model=lambda *args, **kwargs: compiled)
    retries = []
    class CPU:
        def run(self, names, feeds):
            retries.append(feeds['x'].copy())
            return [feeds['x']*2]
    session = accelerators.IntelFlowSession('model.onnx', lambda message: None, lambda path: CPU(), SimpleNamespace(Core=lambda: core))
    value = np.array([3.0], dtype=np.float32)
    first = session.run(None, {'x':value})[0]
    second = session.run(None, {'x':value})[0]
    assert first.tolist() == [1.0]
    assert second.tolist() == [6.0]
    assert retries[0].tolist() == value.tolist() == [3.0]
    assert session.backend == 'onnxruntime-cpu-after-intel-fallback'
    assert session.run(None, {'x':value})[0].tolist() == [6.0]


def test_shape_warmup_repeats_identical_inputs_and_advances_with_only_last_output():
    calls = []
    class Flow:
        backend = 'openvino-intel-gpu-fp32'
        def run(self, names, feeds):
            calls.append(feeds['x'].copy())
            return [np.full_like(feeds['x'], len(calls))]
    session = accelerators.WarmShapeFlow(Flow())
    value = np.array([[3.0, 4.0]], dtype=np.float32)
    first = session.run(None, {'x':value})[0]
    assert first.tolist() == [[3.0, 3.0]]
    assert all(np.array_equal(item, value) for item in calls)
    assert value.tolist() == [[3.0, 4.0]]
    assert session.run(None, {'x':value})[0].tolist() == [[4.0, 4.0]]
    assert session.run(None, {'x':np.ones((1, 3), dtype=np.float32)})[0].tolist() == [[7.0]*3]
    assert session.extra_evaluations == 4
    assert session.shapes == {(1, 2), (1, 3)}


@pytest.mark.parametrize('initial_backend,fallback_call,expected_calls', [
    ('onnxruntime-cpu', None, 1), ('openvino-intel-gpu-fp32', 1, 1),
    ('openvino-intel-gpu-fp32', 2, 2)])
def test_shape_warmup_skips_cpu_and_returns_the_same_step_after_fallback(initial_backend, fallback_call, expected_calls):
    class Flow:
        backend = initial_backend
        calls = 0
        def run(self, names, feeds):
            self.calls += 1
            if self.calls == fallback_call:
                self.backend = 'onnxruntime-cpu-after-intel-fallback'
            return [feeds['x']*2]
    flow = Flow()
    session = accelerators.WarmShapeFlow(flow)
    value = np.array([3.0], dtype=np.float32)
    assert session.run(None, {'x':value})[0].tolist() == [6.0]
    assert flow.calls == expected_calls
    assert session.backend == flow.backend
    assert value.tolist() == [3.0]
