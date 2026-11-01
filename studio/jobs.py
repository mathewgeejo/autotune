"""One job at a time, with a warm isolated worker and process-tree cancellation."""
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
from studio.processes import worker_options, stop_tree


class BusyError(ValueError):
    pass


class JobManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.active = None
        self.jobs = {}
        self.worker = None
        self.engine_log = None
        atexit.register(self.close)

    def _stop_worker(self):
        if self.worker is not None:
            stop_tree(self.worker)
            if self.worker.stdin:
                self.worker.stdin.close()
            self.worker = None
        if self.engine_log:
            self.engine_log.close()
            self.engine_log = None

    def _ensure_worker(self):
        if self.worker is not None and self.worker.poll() is None:
            return
        self._stop_worker()
        self.engine_log = (DATA / 'engine.log').open('a', encoding='utf-8')
        self.worker = subprocess.Popen([sys.executable, '-u', '-m', 'studio.worker', '--serve'],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=self.engine_log, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', **worker_options())

    def submit(self, mode, payload):
        if mode != 'clone':
            raise ValueError('Choose voice cloning to generate speech from text.')
        from studio.speech import split_text
        split_text(payload.get('text'))
        with self.lock:
            if self.active:
                raise BusyError('Another generation is running. Wait for it to finish or cancel it first.')
            payload = dict(payload)
            source = Path(payload.get('source') or '')
            if not source.is_file():
                raise ValueError('Record or upload a voice sample first.')
            if source.stat().st_size > 150 * 1024**2:
                raise ValueError('Upload a voice sample smaller than 150 MB.')
            job_id = uuid.uuid4().hex
            folder = JOBS / job_id
            folder.mkdir(parents=True)
            target = folder / ('source' + source.suffix)
            try:
                shutil.copy2(source, target)
                payload['source'] = str(target)
                request = folder / 'request.json'
                request.write_text(json.dumps({'mode': mode, **payload}), encoding='utf-8')
                (folder / 'worker.log').touch()
                self._ensure_worker()
                job = {'id': job_id, 'mode': mode, 'folder': folder, 'process': self.worker,
                       'started': time.monotonic(), 'state': 'running', 'peak_mb': 0,
                       'cancelled': False}
                self.jobs[job_id] = job
                self.active = job_id
                self.worker.stdin.write(json.dumps(str(request)) + '\n')
                self.worker.stdin.flush()
            except Exception:
                self.active = None
                self.jobs.pop(job_id, None)
                self._stop_worker()
                shutil.rmtree(folder)
                raise
            threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
            return job_id

    def _watch(self, job_id):
        job = self.jobs[job_id]
        process = job['process']
        folder = job['folder']
        while process.poll() is None:
            if (folder / 'result.json').is_file() or (folder / 'failure.json').is_file():
                break
            try:
                parent = psutil.Process(process.pid)
                memory = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])
                job['peak_mb'] = max(job['peak_mb'], memory / 1024**2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.25)
        with self.lock:
            job['elapsed'] = time.monotonic() - job['started']
            try:
                job['result'] = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                job['result'] = {}
            job['state'] = ('cancelled' if job['cancelled'] else
                            'complete' if job['result'] else 'failed')
            if job['state'] == 'failed' and not (folder / 'failure.json').exists():
                (folder / 'status.json').write_text(json.dumps({'stage':
                    'The speech worker stopped unexpectedly. Retry, or check data/engine.log.'}), encoding='utf-8')
            if self.active == job_id:
                self.active = None
            (folder / 'metrics.json').write_text(json.dumps({
                'elapsed_seconds': job['elapsed'], 'peak_memory_mb': job['peak_mb'],
                'state': job['state']}), encoding='utf-8')

    def cancel(self):
        with self.lock:
            if not self.active:
                return 'No generation is running.'
            job = self.jobs[self.active]
            job['cancelled'] = True
            self._stop_worker()
            return 'Cancelled. The speech model has been released from memory.'

    def release(self):
        with self.lock:
            if self.active:
                raise BusyError('Finish or cancel the generation before releasing the model.')
            self._stop_worker()
            return 'Speech model released. The next generation will load it again.'

    def close(self):
        with self.lock:
            if self.active:
                self.jobs[self.active]['cancelled'] = True
            self._stop_worker()

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            try:
                status = json.loads((job['folder'] / 'status.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                status = {'stage': 'Starting local voice cloning'}
            return {**job, 'status': status,
                    'elapsed': job.get('elapsed', time.monotonic()-job['started'])}

    def clear(self, references=False):
        with self.lock:
            if self.active:
                raise BusyError('Finish or cancel the generation before clearing audio.')
            folders = [REFERENCES] if references else [JOBS, DATA / 'tmp']
            # Validate every deletion target before touching any files.
            resolved_folders = [folder.resolve() for folder in folders]
            if any(folder.parent != DATA.resolve() for folder in resolved_folders):
                raise ValueError('Invalid storage directory.')
            if references:
                self._stop_worker()
            for resolved in resolved_folders:
                try:
                    if resolved.exists():
                        shutil.rmtree(resolved)
                except OSError as exc:
                    raise BusyError('Audio is still in use. Stop playback and retry deleting it.') from exc
                resolved.mkdir(parents=True, exist_ok=True)
            if not references:
                self.jobs.clear()
            return 'Cached voice data deleted.' if references else 'Recordings and exports deleted.'
