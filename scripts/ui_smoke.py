"""Browser upload -> actual instrument worker -> playable WAV and MIDI.

Uses the installed Chrome browser. Optional browser driver:
  python -m pip install --target .tools/browser playwright==1.56.0
Run with the studio already listening on localhost:7860.
"""
from pathlib import Path
import sys
import time
import urllib.request
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    source = ROOT / 'data/smoke/melody.wav'
    if not source.exists():
        raise SystemExit('Run scripts/smoke.py instrument first to create the test melody.')
    upload = source.with_name(f'browser-melody-{time.time_ns()}.wav')
    shutil.copy2(source, upload)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 120
    while True:
        try:
            with opener.open('http://127.0.0.1:7860/config', timeout=5) as response:
                assert response.status == 200
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe')
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:7860', wait_until='domcontentloaded', timeout=120000)
        page.get_by_role('heading', name='Your voice. A new possibility.').wait_for()
        page.get_by_role('button', name='Upload file', exact=True).click()
        with page.expect_response(lambda response: '/upload' in response.url and response.request.method == 'POST', timeout=60000):
            page.locator('input[type=file]').first.set_input_files(str(upload))
        page.get_by_text('0:05', exact=True).first.wait_for(timeout=60000)
        page.screenshot(path=str(ROOT / 'data/smoke/studio-upload.png'), full_page=True)
        page.get_by_role('button', name='Transform my voice').click()
        page.wait_for_function("""() => {
          const el = document.querySelector('#status textarea');
          return el && /COMPLETE|FAILED/.test(el.value);
        }""", timeout=240000)
        status = page.locator('#status textarea').input_value()
        print(status)
        assert 'COMPLETE' in status, status
        assert page.locator('a[href*="instrument.wav"]').count() >= 1, 'Missing WAV export'
        assert page.locator('a[href*="melody.mid"]').count() >= 1, 'Missing MIDI export'
        page.screenshot(path=str(ROOT / 'data/smoke/studio-desktop.png'), full_page=True)
        page.get_by_role('tab', name='Singer', exact=True).click()
        page.get_by_label('Reference song link', exact=True).wait_for()
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(ROOT / 'data/smoke/studio-mobile.png'), full_page=True)
        assert not errors, errors
        print('Browser upload, processing, playback controls, exports, Singer tab, and responsive render passed.')
        browser.close()


if __name__ == '__main__':
    main()
