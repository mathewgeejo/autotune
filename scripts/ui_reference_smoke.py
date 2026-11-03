"""Check automatic reference transcription through the running app's browser UI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--port', type=int, default=7862)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--text-file', type=Path, help='Also generate these words and validate the downloaded WAV.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path.')
    report = {'status': 'running', 'page_errors': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True,
                executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe')
            page = browser.new_page(viewport={'width': 1440, 'height': 1100})
            page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
            page.goto(f'http://127.0.0.1:{args.port}', wait_until='domcontentloaded', timeout=120000)
            page.get_by_role('heading', name='Give your words a familiar voice.').wait_for()
            page.get_by_role('button', name='Upload file', exact=True).click()
            page.locator('#voice-sample input[type=file]').set_input_files(str(args.reference.resolve()))
            page.wait_for_function("() => !!document.querySelector('#reference-transcript textarea')?.value.trim()",
                                   timeout=900000)
            page.wait_for_function("() => /COMPLETE|FAILED/.test(document.querySelector('#status textarea').value)",
                                   timeout=900000)
            report['activity'] = page.locator('#status textarea').input_value()
            assert 'COMPLETE' in report['activity'], report['activity']
            full = page.locator('#reference-transcript textarea').input_value()
            page.get_by_text('Selected reference and transcript', exact=True).click()
            selected = page.get_by_label('Words in the selected reference (optional)', exact=True).input_value()
            assert full.strip() and selected.strip()
            report.update(full_word_count=len(full.split()), selected_word_count=len(selected.split()))
            assert len(full.split()) >= len(selected.split())
            page.get_by_role('button', name='Generate speech', exact=True).wait_for(state='visible')
            page.wait_for_function("() => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Generate speech')?.disabled === false")
            if args.text_file:
                expected = args.text_file.read_text(encoding='utf-8')
                jobs = ROOT/'data/jobs'
                before = {path.name for path in jobs.iterdir() if path.is_dir()}
                page.locator('#speech-text textarea').fill(expected)
                page.get_by_role('button', name='Generate speech', exact=True).click()
                page.wait_for_function("() => /Starting|RUNNING/.test(document.querySelector('#status textarea').value)", timeout=30000)
                page.wait_for_function("() => /COMPLETE|FAILED/.test(document.querySelector('#status textarea').value)", timeout=900000)
                report['generation_activity'] = page.locator('#status textarea').input_value()
                assert 'COMPLETE' in report['generation_activity'], report['generation_activity']
                link = page.locator('a[href*="speech.wav"]').first
                link.wait_for(timeout=30000)
                response = page.request.get(urljoin(page.url, link.get_attribute('href')))
                assert response.ok
                digest = hashlib.sha256(response.body()).hexdigest()
                matches = []
                for folder in jobs.iterdir():
                    if folder.name in before or not (folder/'result.json').is_file():
                        continue
                    request = json.loads((folder/'request.json').read_text(encoding='utf-8'))
                    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
                    if request.get('mode') == 'clone' and request.get('text') == expected:
                        if hashlib.sha256(Path(result['audio']).read_bytes()).hexdigest() == digest:
                            matches.append((folder, result))
                assert len(matches) == 1, 'Download must match exactly one new generation of these words.'
                report.update(generation_job=matches[0][0].name, audio_sha256=digest,
                              generated_seconds=matches[0][1]['duration'], passages=matches[0][1]['passages'])
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 2'), 'Mobile overflow'
            assert not report['page_errors'], report['page_errors']
            report['status'] = 'passed'
            browser.close()
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Automatic reference transcription, selected clip words, and mobile layout passed.', flush=True)


if __name__ == '__main__':
    main()
