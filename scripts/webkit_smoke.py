"""Mobile WebKit: upload, text editing, acoustic analysis, exports, and layout."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(ROOT/'.tools/playwright-browsers'))
sys.path.insert(0, str(ROOT/'.tools/browser'))
from playwright.sync_api import sync_playwright


def upload(page, selector, path):
    with page.expect_response(lambda response: '/upload' in response.url and response.request.method == 'POST', timeout=60000):
        page.locator(selector+' input[type=file]').set_input_files(str(path))


def main():
    with sync_playwright() as playwright:
        browser = playwright.webkit.launch(headless=True)
        page = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:7861', wait_until='domcontentloaded', timeout=120000)
        page.get_by_role('heading', name='Give your words a familiar voice.').wait_for()
        page.locator('#speech-text textarea').fill('Hello. A mobile browser can use this interface.')
        assert page.locator('#speech-text textarea').input_value().startswith('Hello')
        page.get_by_role('button', name='Upload file', exact=True).click()
        upload(page, '#voice-sample', ROOT/'data/smoke/dave.wav')
        page.get_by_role('tab', name='Voice analysis', exact=True).click()
        upload(page, '#analysis-reference', ROOT/'data/smoke/dave.wav')
        upload(page, '#analysis-candidate', ROOT/'data/jobs/af007f3489894c92b19173af6f8285ec/speech.wav')
        page.get_by_role('button', name='Analyze voices', exact=True).click()
        page.wait_for_function("() => /COMPLETE|FAILED/.test(document.querySelector('#status textarea').value)", timeout=180000)
        status = page.locator('#status textarea').input_value()
        assert 'COMPLETE' in status, status
        page.locator('.js-plotly-plot').wait_for(timeout=60000)
        page.wait_for_function("""() => {const p=document.querySelector('.js-plotly-plot');
          return p._fullLayout.width <= p.parentElement.clientWidth+2;}""", timeout=10000)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        assert page.locator('a[href*="audio-diagnostics.zip"]').count() >= 1
        page.screenshot(path=str(ROOT/'data/smoke/webkit-mobile.png'), full_page=True)
        report = {'engine': 'WebKit on Windows, mobile viewport (not a physical iPhone)',
                  'upload': True, 'text_input': True, 'acoustic_analysis': True, 'exports': True,
                  'mobile_layout': True, 'errors': errors}
        (ROOT/'data/smoke/webkit-checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))
        assert not errors, errors
        browser.close()


if __name__ == '__main__':
    main()
