"""Verify browser microphone recording with a fake device, then actual conversion.

Requires the same optional Playwright install as ui_smoke.py and a running studio.
No physical microphone is accessed.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    audio = ROOT / 'data/smoke/melody.wav'
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe', args=[
                '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
                f'--use-file-for-fake-audio-capture={audio}'])
        context = browser.new_context(permissions=['microphone'])
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:7860', wait_until='domcontentloaded', timeout=60000)
        page.get_by_role('button', name='Record', exact=True).click()
        page.wait_for_timeout(4500)
        with page.expect_response(lambda r: '/upload' in r.url and r.request.method == 'POST', timeout=60000):
            page.get_by_role('button', name='Stop', exact=True).click()
        page.wait_for_timeout(2000)
        page.get_by_role('button', name='Transform my voice').click()
        page.wait_for_function('''() => {
            const el = document.querySelector('#status textarea');
            return el && /COMPLETE|FAILED/.test(el.value);
        }''', timeout=240000)
        status = page.locator('#status textarea').input_value()
        print(json.dumps({'status': status, 'errors': errors}), flush=True)
        page.screenshot(path=str(ROOT / 'data/smoke/studio-microphone.png'), full_page=True)
        assert 'COMPLETE' in status, status
        assert not errors, errors
        browser.close()


if __name__ == '__main__':
    main()
