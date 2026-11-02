from types import SimpleNamespace

import numpy as np

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


def test_gpu_output_is_owned_and_failed_step_retries_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(accelerators, 'MODELS', tmp_path)
    outputs = np.array([1.0], dtype=np.float32)
    class Request:
        calls = 0
        def infer(self, feeds):
            self.calls += 1
            if self.calls == 2:
                outputs[:] = 999
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
