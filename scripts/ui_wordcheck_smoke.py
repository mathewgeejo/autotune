"""Validate the optional Malayalam word check through an actual browser request.

Requires installed Chrome, .tools/browser, and a running local app. Uses a real
previously measured no-error case; this is workflow coverage, not human listening.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7862)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    if baseline['language'] != 'ml':
        parser.error('Use a measured Malayalam case.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'baseline_report': str(args.baseline_report),
              'scope': 'Browser submission, selected output and no-retry check path. No listening claim.',
              'page_errors': []}
    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    save()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe')
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        try:
            page.goto(f'http://127.0.0.1:{args.port}', wait_until='domcontentloaded', timeout=120000)
            page.get_by_role('heading', name='Give your words a familiar voice.').wait_for()
            assert page.get_by_role('radio', name='Detailed', exact=True).is_checked()
            page.get_by_role('button', name='Upload file', exact=True).click()
            with page.expect_response(lambda r: '/upload' in r.url and r.request.method == 'POST', timeout=60000):
                page.locator('#voice-sample input[type=file]').set_input_files(baseline['source_reference'])
            page.wait_for_function("""() => document.querySelector('#reference-transcript textarea')?.value.trim()
                && [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Generate speech')?.disabled === false""",
                timeout=900000)
            page.get_by_text('Selected reference and transcript', exact=True).click()
            page.get_by_text('Conditioning uses', exact=False).first.wait_for(timeout=60000)
            page.get_by_label('Words in the selected reference (optional)', exact=True).fill(
                baseline['result']['reference_text'])
            page.locator('#speech-text textarea').fill(baseline['expected'])
            page.get_by_role('radio', name='Detailed + word check', exact=True).check()
            jobs_root = ROOT/'data/jobs'
            existing_jobs = {path.name for path in jobs_root.iterdir() if path.is_dir()}
            page.get_by_role('button', name='Generate speech', exact=True).click()
            page.wait_for_function("() => /Starting|RUNNING/.test(document.querySelector('#status textarea').value)", timeout=30000)
            page.wait_for_function("() => /COMPLETE|FAILED/.test(document.querySelector('#status textarea').value)", timeout=1800000)
            report['ui_status'] = page.locator('#status textarea').input_value()
            assert 'COMPLETE' in report['ui_status'], report['ui_status']
            assert 'Automatic word check found no differences.' in report['ui_status']
            href = page.locator('a[href*="speech.wav"]').first.get_attribute('href')
            response = page.request.get(urljoin(page.url, href))
            assert response.ok, response.status
            body = response.body()
            assert body[:4] == b'RIFF' and body[8:12] == b'WAVE'
            audio = Path(unquote(href).split('file=', 1)[1].split('?', 1)[0]).resolve()
            assert any(audio.is_relative_to(path.resolve()) for path in (jobs_root, ROOT/'data/tmp'))
            digest = hashlib.sha256(body).hexdigest()
            matches = []
            for job in jobs_root.iterdir():
                if not job.is_dir() or job.name in existing_jobs or not (job/'result.json').is_file():
                    continue
                request = json.loads((job/'request.json').read_text(encoding='utf-8'))
                if request.get('text', '').strip() != baseline['expected'].strip():
                    continue
                result = json.loads((job/'result.json').read_text(encoding='utf-8'))
                selected = Path(result['audio']).resolve()
                selected.relative_to(job.resolve())
                if hashlib.sha256(selected.read_bytes()).hexdigest() == digest:
                    matches.append((job, result))
            assert len(matches) == 1, 'Downloaded WAV must match exactly one new owned job'
            job, result = matches[0]
            assert result['word_check']['status'] == 'word-check-passed'
            assert not result['word_check']['retry_attempted']
            assert result['precision']['guidance_scale'] == 2
            report.update(audio=result['audio'], download_cache_path=str(audio), audio_sha256=digest,
                          job_id=job.name, word_check=result['word_check'])
            page.screenshot(path=str(args.output.with_suffix('.desktop.png')), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(args.output.with_suffix('.mobile.png')), full_page=True)
            report['mobile_fits'] = page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
            assert report['mobile_fits'], 'Mobile overflow'
            assert not report['page_errors'], report['page_errors']
            report['status'] = 'passed'
            print(report['ui_status'], flush=True)
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            page.screenshot(path=str(args.output.with_suffix('.failure.png')), full_page=True)
            raise
        finally:
            save()
            browser.close()


if __name__ == '__main__':
    main()
