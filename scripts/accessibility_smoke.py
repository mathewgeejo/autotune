"""Axe WCAG 2.1 AA checks and keyboard focus on the live desktop/mobile UI."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.tools/browser'))
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args()
    source = ROOT/'.tools/browser/axe.min.js'
    if not source.is_file():
        raise SystemExit('Install axe-core locally under .tools/browser/axe.min.js before running this optional browser check.')
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe')
        page = browser.new_page()
        page.goto(f'http://127.0.0.1:{args.port}', wait_until='domcontentloaded', timeout=120000)
        page.get_by_role('heading', name='Give your words a familiar voice.').wait_for()
        page.wait_for_function('window.gradio_config !== undefined')
        page.add_script_tag(path=str(source))
        for width in (1440, 390):
            page.set_viewport_size({'width': width, 'height': 1000})
            for tab in ('Create speech', 'Voice analysis'):
                page.get_by_role('tab', name=tab, exact=True).click()
                page.wait_for_timeout(750)
                result = page.evaluate("""async () => {
                    const r = await axe.run(document, {runOnly: {type:'tag', values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}});
                    return {violations:r.violations, passes:r.passes.length, incomplete:r.incomplete.map(x=>({id:x.id,impact:x.impact,nodes:x.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))};
                }""")
                results.append({'width': width, 'tab': tab, **result})
        page.get_by_role('tab', name='Create speech', exact=True).click()
        page.locator('#speech-text textarea').focus()
        page.keyboard.press('Tab')
        focus = page.evaluate("""() => {const e=document.activeElement; const s=getComputedStyle(e); return {tag:e.tagName,text:e.textContent,outline:s.outline,shadow:s.boxShadow};}""")
        print(json.dumps({'keyboard_after_text': focus}, ensure_ascii=False))
        assert page.get_by_role('button', name='Generate speech', exact=True).evaluate('(e)=>e===document.activeElement')
        browser.close()
    output = ROOT/'data/smoke/accessibility.json'
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps({'width':result['width'],'tab':result['tab'],'passes':result['passes'],
                          'violations':[(v['id'],len(v['nodes'])) for v in result['violations']]}))
    assert not any(result['violations'] for result in results), f'Accessibility violations: {output}'


if __name__ == '__main__':
    main()
