"""Browser upload -> actual cloned speech -> WAV, warm second generation.

Requires optional Playwright under .tools/browser and installed Chrome.
Run app.py --no-browser --port 7861 first.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--reference', type=Path, default=ROOT / 'data/smoke/dave.wav')
    parser.add_argument('--layout-only', action='store_true')
    parser.add_argument('--long-text', action='store_true')
    args = parser.parse_args()
    screenshots = ROOT / 'data/smoke'
    screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe')
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        failed_requests = []
        page.on('pageerror', lambda error: errors.append(error.stack or str(error)))
        page.on('requestfailed', lambda request: failed_requests.append((request.url, request.failure)))
        page.goto(f'http://127.0.0.1:{args.port}', wait_until='domcontentloaded', timeout=120000)
        page.get_by_role('heading', name='Give your words a familiar voice.').wait_for()
        if not args.layout_only:
            page.get_by_text('Local storage & memory', exact=True).click()
            page.get_by_role('button', name='Release model from memory', exact=True).click()
            page.get_by_text('Local storage & memory', exact=True).click()
        page.get_by_role('button', name='Upload file', exact=True).click()
        with page.expect_response(lambda r: '/upload' in r.url and r.request.method == 'POST', timeout=60000):
            page.locator('#voice-sample input[type=file]').set_input_files(str(args.reference))
        page.locator('#speech-text textarea').fill('Hello there. These words are generated in the voice from my sample.')
        page.screenshot(path=str(screenshots / 'cloning-desktop.png'), full_page=True)
        if not args.layout_only:
            first_text = 'Hello there. These words are generated in the voice from my sample.'
            if args.long_text:
                first_text = ('Good morning. I have a few things to share with you today. '
                    'The meeting starts at nine, and the blue notebook is on the table. '
                    'Please bring it with you when you come to the office. We will read the plan '
                    'together and decide what to do next. Thank you for your help. '
                    'The final words are see you soon.')
            for index, text in enumerate([
                first_text,
                'Here is a different sentence. The same voice can read new words.']):
                page.locator('#speech-text textarea').fill(text)
                page.get_by_role('button', name='Generate speech', exact=True).click()
                page.wait_for_function("() => /Starting|RUNNING/.test(document.querySelector('#status textarea').value)",
                                       timeout=30000)
                if args.long_text and index == 0:
                    page.locator('#speech-output a[href*="passage-1.wav"]').first.wait_for(timeout=900000)
                    print('First passage playable while remaining speech is generated.', flush=True)
                page.wait_for_function("() => /COMPLETE|FAILED/.test(document.querySelector('#status textarea').value)",
                                       timeout=900000)
                status = page.locator('#status textarea').input_value()
                print(status, flush=True)
                assert 'COMPLETE' in status, status
                assert page.locator('a[href*="speech.wav"]').count() >= 1, 'Missing WAV export'
                if index:
                    assert 'cached voice reused' in status
            page.screenshot(path=str(screenshots / 'cloning-result.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(screenshots / 'cloning-mobile.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2'), 'Mobile overflow'
        assert not errors, {'errors': errors, 'failed_requests': failed_requests}
        print('Browser voice-sample upload, text input, responsive layout, and requested generation checks passed.')
        browser.close()


if __name__ == '__main__':
    main()
