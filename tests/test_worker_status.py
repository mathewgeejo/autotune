import json
from pathlib import Path

from studio import worker


def test_atomic_status_retries_a_brief_reader_lock(tmp_path, monkeypatch):
    target = tmp_path/'status.json'
    target.write_text('{"stage":"previous"}', encoding='utf-8')
    original = Path.replace
    attempts = []
    def replace(path, destination):
        attempts.append(path)
        if len(attempts) < 3:
            assert json.loads(target.read_text())['stage'] == 'previous'
            raise PermissionError('Simulated Windows reader lock')
        return original(path, destination)
    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(worker.time, 'sleep', lambda seconds: None)
    worker.atomic_json(target, {'stage':'complete'})
    assert len(attempts) == 3
    assert json.loads(target.read_text()) == {'stage':'complete'}


def test_progress_file_lock_does_not_fail_speech_request(tmp_path, monkeypatch):
    from studio import speech
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'mode':'clone','source':'fixture.wav','text':'Hello there.','language':'en'}))
    monkeypatch.setattr(speech, 'prepare_sample', lambda *args, **kwargs: (None, tmp_path/'reference.wav', 'digest'))
    write = worker.atomic_json
    def locked_status(path, value):
        if path.name == 'status.json':
            raise PermissionError('Simulated sustained status lock')
        write(path, value)
    monkeypatch.setattr(worker, 'atomic_json', locked_status)
    class Engine:
        engine_key = ('en',)
        def synthesize(self, request, folder, progress, prepared=None):
            progress('Rendering the speech waveform')
            return {'audio':'completed.wav'}
    engine = Engine()
    assert worker.run(request, engine) is engine
    assert json.loads((tmp_path/'result.json').read_text()) == {'audio':'completed.wav'}
    assert not (tmp_path/'failure.json').exists()
