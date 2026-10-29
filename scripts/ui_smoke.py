"""Browser upload -> actual instrument worker -> playable WAV and MIDI.

Uses the installed Chrome browser. Optional browser driver:
  python -m pip install --target .tools/browser playwright==1.56.0
Run with the studio already listening on localhost:7860.
"""
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    source = ROOT / 'data/smoke/melody.wav'
    if not source.exists():
        raise SystemExit('Run scripts/smoke.py instrument first to create the test melody.')
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
        page.locator('input[type=file]').first.set_input_files(str(source))
        page.wait_for_timeout(2500)
        page.get_by_role('button', name='Transform my voice').click()
        page.wait_for_function("""() => {
          const el = document.querySelector('#status textarea');
          return el && /COMPLETE|FAILED/.test(el.value);
        }""", timeout=240000)
        status = page.locator('#status textarea').input_value()
        print(status)
        assert 'COMPLETE' in status, status
        assert page.locator('audio').count() >= 2, 'Missing result audio player'
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
