import subprocess
import sys
import time

import psutil
import pytest

import studio.jobs as jobs
from studio.paths import configure


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows process-tree cancellation')
def test_cancel_kills_children_and_allows_next_task(tmp_path, monkeypatch):
    configure()
    monkeypatch.setattr(jobs, 'JOBS', tmp_path)
    popen = subprocess.Popen
    # Replace only the audio worker with a long-running parent and child process.
    child_pid_file = tmp_path / 'child.pid'
    code = (
        'import subprocess,sys,time,pathlib; '
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
        f'pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); '
        'time.sleep(60)'
    )
    monkeypatch.setattr(jobs.subprocess, 'Popen', lambda args, **kwargs:
                        popen([sys.executable, '-c', code], **kwargs) if '-m' in args else popen(args, **kwargs))
    manager = jobs.JobManager()
    first = manager.submit('instrument', {})
    try:
        with pytest.raises(jobs.BusyError):
            manager.submit('singer', {})
        deadline = time.monotonic() + 10
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        child = int(child_pid_file.read_text())
        # A cancellation during result writing must still release the job slot.
        (tmp_path/first/'result.json').write_text('{', encoding='utf-8')
        manager.cancel()
        while manager.active and time.monotonic() < deadline:
            time.sleep(0.1)
        assert manager.snapshot(first)['state'] == 'cancelled'
        assert not psutil.pid_exists(child)
        second = manager.submit('instrument', {})
        assert second != first
    finally:
        manager.cancel()


def test_storage_cleanup_preserves_models_and_rejects_outside_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'DATA', tmp_path)
    monkeypatch.setattr(jobs, 'JOBS', tmp_path/'jobs')
    monkeypatch.setattr(jobs, 'REFERENCES', tmp_path/'references')
    for folder in ('jobs', 'references', 'models', 'tmp'):
        target = tmp_path/folder
        target.mkdir()
        (target/'audio-or-model.bin').write_bytes(b'content')
    manager = jobs.JobManager()
    manager.clear()
    assert not list((tmp_path/'jobs').iterdir())
    assert (tmp_path/'references/audio-or-model.bin').exists()
    manager.clear(references=True)
    assert not list((tmp_path/'references').iterdir())
    assert (tmp_path/'models/audio-or-model.bin').exists()
    monkeypatch.setattr(jobs, 'JOBS', tmp_path.parent)
    with pytest.raises(ValueError, match='Invalid storage'):
        manager.clear()
