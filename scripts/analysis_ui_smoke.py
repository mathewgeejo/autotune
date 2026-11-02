"""Actual browser comparison, chart rendering, exports, and device layouts."""
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic()+120
    while True:
        try:
            opener.open('http://127.0.0.1:7861/config', timeout=5).close()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe')
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:7861', wait_until='domcontentloaded', timeout=120000)
        page.get_by_role('tab', name='Voice analysis', exact=True).click()
        paths = [ROOT/'data/smoke/dave.wav', ROOT/'data/jobs/af007f3489894c92b19173af6f8285ec/speech.wav']
        for index, path in enumerate(paths):
            control = page.locator('#analysis-reference' if index == 0 else '#analysis-candidate')
            with page.expect_response(lambda response: '/upload' in response.url and response.request.method == 'POST', timeout=60000):
                control.locator('input[type=file]').set_input_files(str(path))
        page.get_by_role('button', name='Analyze voices', exact=True).click()
        page.wait_for_function("() => /Starting|RUNNING/.test(document.querySelector('#status textarea').value)", timeout=30000)
        page.wait_for_function("() => /COMPLETE|FAILED/.test(document.querySelector('#status textarea').value)", timeout=180000)
        status = page.locator('#status textarea').input_value()
        print(status, flush=True)
        assert 'COMPLETE' in status, status
        page.get_by_text('Independent speaker cosine:', exact=False).count()
        assert page.locator('a[href*="analysis.json"]').count() >= 1
        assert page.locator('a[href*="analysis-frames.csv"]').count() >= 1
        assert page.locator('a[href*="audio-diagnostics.zip"]').count() >= 1
        page.locator('.js-plotly-plot').wait_for(timeout=60000)
        assert page.evaluate("document.querySelector('.js-plotly-plot').data.length") == 6
        for width in (1440, 768, 390, 360):
            page.set_viewport_size({'width': width, 'height': 1000 if width > 640 else 844})
            # Gradio/Plotly's ResizeObserver updates after the browser resize;
            # wait for the responsive layout rather than photographing a stale chart.
            try:
                page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth + 2', timeout=10000)
            except Exception:
                print(page.evaluate("""() => [...document.querySelectorAll('body *')].filter(e =>
                  e.offsetParent && e.getBoundingClientRect().right > innerWidth + 2).sort((a,b) => b.getBoundingClientRect().width-a.getBoundingClientRect().width).slice(-30).map(e =>
                  ({tag:e.tagName,id:e.id,cls:e.className,width:e.getBoundingClientRect().width,
                    right:e.getBoundingClientRect().right}))"""), flush=True)
                page.screenshot(path=str(ROOT/f'data/smoke/analysis-overflow-{width}.png'), full_page=True)
                raise
            page.screenshot(path=str(ROOT/f'data/smoke/analysis-{width}.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2'), f'Overflow at {width}px'
        assert not errors, errors
        browser.close()
        print('Actual acoustic comparison, six plot traces, JSON/CSV/ZIP downloads, and four layouts passed.', flush=True)


if __name__ == '__main__':
    main()
