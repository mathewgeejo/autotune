"""Verify instance login and file boundaries using an ephemeral test password."""
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, JOBS
configure()
from studio.processes import worker_options, stop_tree
import httpx


def main():
    env = os.environ.copy()
    env.update(VOICE_STUDIO_USERNAME='verification', VOICE_STUDIO_PASSWORD=secrets.token_urlsafe(32),
               VOICE_STUDIO_REQUIRE_AUTH='1')
    port = 7862
    process = subprocess.Popen([sys.executable, '-u', 'app.py', '--no-browser', '--port', str(port)],
                               cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               **worker_options())
    checks = {}
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=10) as client:
            deadline = time.monotonic()+90
            while True:
                try:
                    response = client.get('/config')
                    break
                except httpx.TransportError:
                    if time.monotonic() > deadline or process.poll() is not None:
                        raise
                    time.sleep(0.5)
            assert response.status_code == 401, response.status_code
            checks['configuration_requires_login'] = True
            assert client.post('/login', data={'username': 'verification', 'password': 'incorrect'}).status_code == 400
            checks['incorrect_password_rejected'] = True
            login = client.post('/login', data={'username': env['VOICE_STUDIO_USERNAME'],
                                               'password': env['VOICE_STUDIO_PASSWORD']})
            assert login.status_code == 200
            assert client.get('/config').status_code == 200
            checks['authenticated_configuration_available'] = True
            path = ROOT/'README.md'
            assert client.get('/gradio_api/file='+str(path)).status_code == 403
            checks['arbitrary_project_files_forbidden'] = True
            import numpy as np
            import soundfile as sf
            folder = JOBS/('auth-check-'+secrets.token_hex(8))
            folder.mkdir()
            audio = folder/'authorized.wav'
            sf.write(audio, np.zeros(2400), 24000)
            assert client.get('/gradio_api/file='+str(audio)).status_code == 200
            checks['authorized_output_downloads'] = True
            client.cookies.clear()
            assert client.get('/gradio_api/file='+str(audio)).status_code == 401
            checks['output_download_requires_login'] = True
        (ROOT/'data/smoke/auth-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        print(json.dumps(checks, indent=2))
    finally:
        stop_tree(process)
        if process.stdout:
            process.stdout.close()


if __name__ == '__main__':
    main()
