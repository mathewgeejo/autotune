"""Resumable, bounded public-checkpoint download for the isolated F5 experiment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, MODELS
configure()
from studio.runtime import keep_awake
import requests

REVISION = 'b169ad1af92d659f9b85ac393510f8960e4a0690'
URL = f'https://huggingface.co/dheeyantra/dhee-indic-f5/resolve/{REVISION}/model.safetensors'
SIZE = 1402784904
SHA256 = '23d5cd75f5b52b54bd04efe39f871900146fa7123ae2c42fe5a38d04c7025738'
CHUNK = 4*1024*1024


def fetch(index):
    start, end = index*CHUNK, min((index+1)*CHUNK, SIZE)-1
    expected = end-start+1
    for attempt in range(5):
        try:
            with requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, stream=True,
                              timeout=(15, 30)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('The server did not return the exact bounded byte range')
                body = response.raw.read(expected+1)
                if len(body) != expected:
                    raise ValueError('The bounded byte range is incomplete')
                return index, body
        except (requests.RequestException, OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(4, 0.5*2**attempt))


def main():
    folder = MODELS/'experimental-f5'/REVISION
    folder.mkdir(parents=True, exist_ok=True)
    destination = (folder/'model.safetensors').resolve()
    partial = destination.with_suffix('.part')
    manifest = folder/'download.json'
    if not destination.is_relative_to(MODELS.resolve()) or partial.parent != destination.parent:
        raise RuntimeError('The model output must stay inside the configured model directory')
    if destination.exists():
        raise RuntimeError('The completed checkpoint already exists; preserve the prior result')
    state = {'model': 'dheeyantra/dhee-indic-f5', 'revision': REVISION, 'source': URL,
             'expected_bytes': SIZE, 'expected_sha256': SHA256, 'chunk_bytes': CHUNK,
             'completed': {}, 'status': 'downloading'}
    if manifest.exists():
        state = json.loads(manifest.read_text(encoding='utf-8'))
        if state['expected_sha256'] != SHA256 or state['chunk_bytes'] != CHUNK:
            raise RuntimeError('Existing partial download belongs to a different checkpoint')
    def save():
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(manifest)
    with keep_awake(), partial.open('r+b' if partial.exists() else 'w+b') as output:
        output.truncate(SIZE)
        # Verify completed ranges before trusting a resumed manifest.
        for key, digest in list(state['completed'].items()):
            index = int(key)
            output.seek(index*CHUNK)
            body = output.read(min(CHUNK, SIZE-index*CHUNK))
            if hashlib.sha256(body).hexdigest() != digest:
                del state['completed'][key]
        probe = ROOT/'.tools/dhee-range-probe.bin'
        if '0' not in state['completed'] and probe.is_file() and probe.stat().st_size == CHUNK:
            body = probe.read_bytes()
            output.seek(0)
            output.write(body)
            state['completed']['0'] = hashlib.sha256(body).hexdigest()
        output.flush()
        save()
        count = (SIZE+CHUNK-1)//CHUNK
        pending = [index for index in range(count) if str(index) not in state['completed']]
        print(f'Checkpoint download: {len(state["completed"])}/{count} ranges verified', flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, index) for index in pending]
            for future in as_completed(futures):
                try:
                    index, body = future.result()
                except Exception as exc:
                    state.update(status='download-failed', error=f'{type(exc).__name__}: {exc}')
                    save()
                    for item in futures:
                        item.cancel()
                    raise
                output.seek(index*CHUNK)
                output.write(body)
                output.flush()
                state['completed'][str(index)] = hashlib.sha256(body).hexdigest()
                state['downloaded_bytes'] = sum(min(CHUNK, SIZE-int(key)*CHUNK) for key in state['completed'])
                save()
                print(f'Verified ranges {len(state["completed"])}/{count}; {state["downloaded_bytes"]/1024**2:.1f} MiB', flush=True)
        output.seek(0)
        digest = hashlib.sha256()
        for body in iter(lambda: output.read(CHUNK), b''):
            digest.update(body)
        actual = digest.hexdigest()
        state['sha256'] = actual
        if actual != SHA256:
            state['status'] = 'checksum-failed'
            save()
            raise RuntimeError('Full checkpoint checksum does not match the published model hash')
    partial.replace(destination)
    state.update(status='verified', path=str(destination), bytes=destination.stat().st_size)
    state.pop('error', None)
    save()
    print('Verified checkpoint:', destination, flush=True)


if __name__ == '__main__':
    main()
