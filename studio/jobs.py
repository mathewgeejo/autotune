"""One cancellable process tree at a time; AI memory is freed after each job."""
import atexit
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

import psutil
from studio.paths import JOBS, REFERENCES, DATA, ROOT


class BusyError(ValueError):
    pass


class JobManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.active = None
        self.jobs = {}
        atexit.register(self.cancel)

    def submit(self, mode, payload):
        with self.lock:
            if self.active:
                raise BusyError('Another task is running. Wait for it to finish or cancel it first.')
            job_id = uuid.uuid4().hex
            folder = JOBS / job_id
            folder.mkdir(parents=True)
            # Own inputs so clearing/replacing the UI recording cannot invalidate a running job.
            for key in ('source', 'file'):
                if payload.get(key):
                    source = Path(payload[key])
                    if not source.is_file():
                        raise ValueError('The selected audio file is missing. Please select it again.')
                    target = folder / (key + source.suffix)
                    shutil.copy2(source, target)
                    payload[key] = str(target)
            request = folder / 'request.json'
            request.write_text(json.dumps({'mode': mode, **payload}), encoding='utf-8')
            log = (folder / 'worker.log').open('w', encoding='utf-8')
            try:
                process = subprocess.Popen([sys.executable, '-u', '-m', 'studio.worker', str(request)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=0x08000000)
            except Exception:
                log.close()
                raise
            self.jobs[job_id] = {'id': job_id, 'mode': mode, 'folder': folder, 'process': process,
                                 'started': time.monotonic(), 'state': 'running', 'peak_mb': 0,
                                 'log': log, 'cancelled': False}
            self.active = job_id
            threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
            return job_id

    def _watch(self, job_id):
        job = self.jobs[job_id]
        process = job['process']
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                memory = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])
                job['peak_mb'] = max(job['peak_mb'], memory / 1024**2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.5)
        with self.lock:
            job['log'].close()
            job['elapsed'] = time.monotonic() - job['started']
            result_path = job['folder'] / 'result.json'
            job['result'] = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
            job['state'] = ('cancelled' if job['cancelled'] else
                            'complete' if process.returncode == 0 and job['result'] else 'failed')
            if self.active == job_id:
                self.active = None
            (job['folder'] / 'metrics.json').write_text(json.dumps({
                'elapsed_seconds': job['elapsed'], 'peak_memory_mb': job['peak_mb'],
                'state': job['state']}), encoding='utf-8')

    def cancel(self):
        with self.lock:
            if not self.active:
                return 'No task is running.'
            job = self.jobs[self.active]
            if job['process'].poll() is not None:
                return 'Task finished.'
            job['cancelled'] = True
            # taskkill /T is necessary: Demucs/ffmpeg are children of the worker.
            subprocess.run(['taskkill', '/PID', str(job['process'].pid), '/T', '/F'],
                           capture_output=True, check=False, creationflags=0x08000000)
            return 'Cancelling… Audio models will be released when the worker exits.'

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            status_path = job['folder'] / 'status.json'
            try:
                status = json.loads(status_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                status = {'stage': 'Starting local audio engine'}
            return {**job, 'status': status,
                    'elapsed': job.get('elapsed', time.monotonic()-job['started'])}

    def clear(self, references=False):
        with self.lock:
            if self.active:
                raise BusyError('Finish or cancel the current task before clearing audio.')
            folders = [REFERENCES] if references else [JOBS, DATA / 'tmp']
            for folder in folders:
                resolved = folder.resolve()
                if resolved.parent != DATA.resolve():
                    raise ValueError('Invalid storage directory.')
                shutil.rmtree(resolved, ignore_errors=True)
                resolved.mkdir(parents=True, exist_ok=True)
            if not references:
                self.jobs.clear()
            return 'Cached references deleted.' if references else 'Recordings and exports deleted.'
